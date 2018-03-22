import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
import math
from graph_encoder import GraphAttentionEncoder
from torch.nn import DataParallel


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


class AttentionModel(nn.Module):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 decode_type="sampling",
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8):
        super(AttentionModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.norm_factor = math.sqrt(embedding_dim)
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = decode_type
        self.temp = 1.0

        self.tanh_clipping = tanh_clipping

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads

        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            node_dim=self.problem.NODE_DIM,
            normalization=normalization
        )

        # TSP specific context parameters (placeholder and step context dimension)
        step_context_dim = 2 * embedding_dim  # Embedding of first and last node
        # Learned input symbols for first action
        std = 1. / math.sqrt(embedding_dim)
        self.W_placeholder = nn.Parameter(torch.Tensor(2 * embedding_dim))
        self.W_placeholder.data.uniform_(-std, std)

        # For each node we comput (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        # self.project_glimpse = nn.Linear(embedding_dim, embedding_dim, bias=False)
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def update_mask(self, mask, selected):
        return mask.clone().scatter_(1, selected.unsqueeze(-1), True)

    def forward(self, input, eval_seq=None):
        """

        :param input: (batch_size, graph_size, node_dim) input node features
        :param eval_seq: (batch_size, graph_size) sequence to score (supervised) or None for autoregressive
        :return:
        """

        embeddings, _ = self.embedder(input)

        log_p, pi = self._inner(embeddings, eval_seq)

        cost, mask = self.problem.get_costs(input, pi)

        return cost, log_p, pi, mask

    def _get_context(self, embeddings, sequences):

        if len(sequences) == 0:
            # First step, use learned input symbol (placeholder)
            # No need to repeat, by adding dimension broadcasting will work
            return self.W_placeholder.unsqueeze(0)
        else:
            batch_size = embeddings.size(0)
            # Return first and last node embeddings
            return torch.gather(
                embeddings,
                1,
                torch.stack((sequences[0], sequences[-1]), dim=1)
                .contiguous()
                .view(batch_size, 2, 1)
                .expand(batch_size, 2, embeddings.size(-1))
            ).view(batch_size, -1)  # View to have (batch_size, 2 * embed_dim)

    def _select_node(self, probs, mask):

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)

            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected

    def _inner(self, embeddings, eval_seq=None):

        batch_size, graph_size, embed_dim = embeddings.size()
        assert embed_dim == self.embedding_dim
        key_size = val_size = self.embedding_dim // self.n_heads

        outputs = []
        sequences = []
        mask = Variable(embeddings.data.new().byte().new(batch_size, graph_size).zero_())

        # The fixed context projection is calculated only once for efficiency
        graph_embed = embeddings.mean(1)
        fixed_context = self.project_fixed_context(graph_embed)

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        glimpse_key, glimpse_val, logit_key = self.project_node_embeddings(embeddings).chunk(3, dim=2)
        # Rearrange dimensions so the dimensions are (n_heads, batch_size, graph_size, key/val_size)
        glimpse_K = glimpse_key.contiguous().view(batch_size, graph_size, self.n_heads, key_size).permute(2, 0, 1, 3)
        glimpse_V = glimpse_val.contiguous().view(batch_size, graph_size, self.n_heads, val_size).permute(2, 0, 1, 3)
        # No need to rearrange key for logit as there is a single head
        logit_K = logit_key.contiguous().view(batch_size, graph_size, self.embedding_dim)

        # Perform decoding steps
        for i in range(graph_size):

            # Compute query = context node embedding
            # We can add the fixed projected context (graph embedding) to the projected step context
            query = fixed_context + self.project_step_context(self._get_context(embeddings, sequences))

            # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, 1, key_size)
            glimpse_Q = query.view(batch_size, 1, self.n_heads, key_size).permute(2, 0, 1, 3)

            # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, 1, graph_size)
            compatibility = self.norm_factor * torch.matmul(glimpse_Q, glimpse_K.transpose(2, 3))
            if self.mask_inner:
                assert self.mask_logits, "Cannot mask inner without masking logits"
                compatibility[mask.view(1, batch_size, 1, graph_size).expand_as(compatibility)] = -math.inf

            # Batch matrix multiplication to compute heads (n_heads, batch_size, 1, val_size)
            heads = torch.matmul(F.softmax(compatibility, dim=-1), glimpse_V)

            # Project to get glimpse/updated context node embedding (batch_size, embedding_dim)
            glimpse = self.project_out(heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * val_size))

            # Now projecting the glimpse is not needed since this can be absorbed into project_out
            # final_Q = self.project_glimpse(glimpse)
            final_Q = glimpse
            # Batch matrix multiplication to compute logits (batch_size, graph_size, 1) => (batch_size, graph_size)
            logits = self.norm_factor * torch.matmul(logit_K, final_Q.unsqueeze(-1)).squeeze(-1)  # 'compatibility'

            # From the logits compute the probabilities by clipping, masking and softmax
            if self.tanh_clipping > 0:
                logits = F.tanh(logits) * self.tanh_clipping
            if self.mask_logits:
                logits[mask] = -math.inf
            log_p = F.log_softmax(logits / self.temp, dim=1)
            probs = log_p.exp()  # (batch_size, graph_size)

            # Select the indices of the next nodes in the sequences (or evaluate eval_seq), result (batch_size) long
            selected = self._select_node(probs, mask) if eval_seq is None else eval_seq[:, i]

            # Update the mask
            mask = self.update_mask(mask, selected)

            # Collect output of step
            outputs.append(log_p)
            sequences.append(selected)

        # Collected lists, return Tensor
        return torch.stack(outputs, 1), torch.stack(sequences, 1)
