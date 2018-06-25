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

    VEHICLE_CAPACITY = 1.  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 fix_norm_factor=False):
        super(AttentionModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.norm_factor = 1 / math.sqrt(embedding_dim) if fix_norm_factor else math.sqrt(embedding_dim)
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0
        self.allow_partial = problem.NAME == 'sdvrp'
        self.is_vrp = problem.NAME == 'cvrp' or problem.NAME == 'sdvrp'

        self.tanh_clipping = tanh_clipping

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads

        # Problem specific context parameters (placeholder and step context dimension)
        if self.is_vrp:
            step_context_dim = embedding_dim + 1  # Embedding of last node + remaining_capacity
            node_dim = 3  # x, y, demand

            # Special embedding projection for depot node
            self.init_embed_depot = nn.Linear(2, embedding_dim)
            
            if self.allow_partial:  # Need to include the demand if split delivery allowed
                self.project_node_step = nn.Linear(1, 3 * embedding_dim, bias=False)
        else:  # TSP
            assert problem.NAME == 'tsp', "Unsupported problem: {}".format(problem.NAME)
            step_context_dim = 2 * embedding_dim  # Embedding of first and last node
            node_dim = 2  # x, y
            
            # Learned input symbols for first action
            self.W_placeholder = nn.Parameter(torch.Tensor(2 * embedding_dim))
            self.W_placeholder.data.uniform_(-1, 1)  # Placeholder should be in range of activations

        self.init_embed = nn.Linear(node_dim, embedding_dim)

        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, input, return_pi=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features
        :return:
        """

        embeddings, _ = self.embedder(self._init_embed(input))

        _log_p, pi = self._inner(input, embeddings)

        cost, mask = self.problem.get_costs(input, pi)
        # Log likelyhood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        ll = self._calc_log_likelihood(_log_p, pi, mask)
        if return_pi:
            return cost, ll, pi

        return cost, ll

    def _calc_log_likelihood(self, _log_p, a, mask):

        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)

    def _init_embed(self, input):

        if self.is_vrp:
            return torch.cat(
                (
                    self.init_embed_depot(input['depot'])[:, None, :],
                    self.init_embed(torch.cat((input['loc'], input['demand'][:, :, None]), -1))
                ),
                1
            )

        return self.init_embed(input)

    def _inner(self, input, embeddings):

        outputs = []
        sequences = []

        state = self._init_state(input)

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed_context, attention_node_data_fixed = self._precompute(embeddings)

        # Perform decoding steps
        i = 0
        while not self._is_finished(i, state):

            log_p, mask = self._get_log_p(embeddings, fixed_context, attention_node_data_fixed, state)

            # Select the indices of the next nodes in the sequences, result (batch_size) long
            selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])  # Squeeze out steps dimension

            state = self._update_state(state, selected)

            # Collect output of step
            outputs.append(log_p[:, 0, :])
            sequences.append(selected)

            i += 1

        # Collected lists, return Tensor
        return torch.stack(outputs, 1), torch.stack(sequences, 1)

    def _is_finished(self, i, state):
        prev_a, demands_with_depot, used_capacity, first_a, mask = state

        if self.is_vrp:
            # For VRP, i >= graph_size is a quick necessary condition
            return i >= demands_with_depot.size(-1) and not (demands_with_depot > 0).any()

        # TSP
        return i >= mask.size(-1)

    def _init_state(self, input):
        loc = input['loc'] if isinstance(input, dict) else input
        batch_size = loc.size(0)

        prev_a = Variable(torch.zeros(batch_size, 1, out=loc.data.new().long().new()))
        first_a = None

        if self.is_vrp:

            demands_with_depot = torch.cat((
                Variable(torch.zeros(batch_size, 1, out=input['demand'].data.new())),
                input['demand'][:, :]
            ), 1)[:, None, :]  # Add steps dimension
            assert (demands_with_depot <= self.VEHICLE_CAPACITY).all(), "Demands must be < vehicle capacity"
            used_capacity = Variable(loc.data.new(batch_size, 1).fill_(0))
            mask = None  # For VRP, mask is implicit via (remaining) demands

        else:  # TSP
            demands_with_depot = used_capacity = None
            # For TSP keep explicit mask
            mask = Variable(loc.data.new().byte().new(loc.size(0), 1, loc.size(1)).zero_())

        # Second dimension is the step which is always a singleton for now
        return prev_a, demands_with_depot, used_capacity, first_a, mask

    def _update_state(self, state, selected):

        prev_a, demands_with_depot, used_capacity, first_a, mask = state

        # Update the state
        prev_a = selected[:, None]  # Add dimension for step

        if self.is_vrp:
            # We don't want to modify in place, prevent trouble
            demands_with_depot = demands_with_depot.clone()[:, 0, :]
            used_capacity = used_capacity.clone()[:, 0]

            # Since most nodes won't visit depot,
            # maybe it is more efficient to do demands.gather(1, torch.max(selected - 1, 0)]
            # as afterwards if selected == 0 (depot) it will be reset anyway
            not_to_depot = torch.nonzero(selected != 0)

            if len(not_to_depot) > 0:
                # Not sure if boolean indexing or numeric is faster
                # If we do not visit the depot, we visit one additional node
                d = demands_with_depot[not_to_depot[:, 0], selected[selected != 0]]
                if self.allow_partial:
                    d = torch.min(d, self.VEHICLE_CAPACITY - used_capacity[selected != 0])
                used_capacity[selected == 0] = 0
                used_capacity[selected != 0] += d
                demands_with_depot[not_to_depot[:, 0], selected[selected != 0]] -= d

            else:
                # All routes visit depot, set all capacities to 0
                used_capacity[:] = 0

            demands_with_depot = demands_with_depot[:, None, :]
            used_capacity = used_capacity[:, None]

        else:  # TSP

            if first_a is None:
                first_a = prev_a
            # Update mask (temporarily squeeze steps dimension)
            mask = mask[:, 0, :].clone().scatter_(1, prev_a, True)[:, None, :]

        return prev_a, demands_with_depot, used_capacity, first_a, mask

    def _select_node(self, probs, mask):

        assert (probs == probs).all(), "Probs should not contain any nans"

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

    def _precompute(self, embeddings, num_steps=1):

        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        fixed_attention_node_data = self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        if not self.allow_partial:

            # If we don't allow partial delivery, the node embeddings are not updated with the remaining demands
            # and we can reshape once for efficiency, otherwise the reshape is done in each step after
            # adding embedding of the remaining demand

            glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = fixed_attention_node_data

            # No need to rearrange key for logit as there is a single head
            fixed_attention_node_data = (
                self._make_heads(glimpse_key_fixed, num_steps),
                self._make_heads(glimpse_val_fixed, num_steps),
                logit_key_fixed.contiguous()
            )

        return fixed_context, fixed_attention_node_data

    def _get_log_p(self, embeddings, fixed_context, attention_node_data_fixed, state):

        prev_a, demands_with_depot, used_capacity, first_a, mask = state

        # Compute query = context node embedding
        query = fixed_context + self.project_step_context(
            self._get_parallel_step_context(embeddings, prev_a, first_a=first_a, used_capacity=used_capacity)
        )
        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(attention_node_data_fixed, demands_with_depot)
        # Compute the mask
        mask = self._get_mask(demands_with_depot, used_capacity, prev_a, mask)
        # Compute logits
        logits, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)
        log_p = F.log_softmax(logits / self.temp, dim=-1)
        return log_p, mask

    def _get_parallel_step_context(self, embeddings, prev_a, first_a=None, used_capacity=None):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)
        
        :param embeddings: (batch_size, graph_size, embed_dim)
        :param prev_a: (batch_size, num_steps)
        :param first_a: Only used when num_steps = 1, action of first step or None if first step
        :return: (batch_size, num_steps, context_dim)
        """

        batch_size, num_steps = prev_a.size()

        if self.is_vrp:
            # Embedding of previous node + remaining capacity
            return torch.cat((
                torch.gather(
                    embeddings,
                    1,
                    prev_a.contiguous()
                        .view(batch_size, num_steps, 1)
                        .expand(batch_size, num_steps, embeddings.size(-1))
                ).view(batch_size, num_steps, embeddings.size(-1)),  # View to have (batch_size, num_steps, embed_dim)
                self.VEHICLE_CAPACITY - used_capacity[:, :, None]
            ), -1)
        else:  # TSP
        
            if num_steps == 1:  # We need to special case if we have only 1 step, may be the first or not
                if first_a is None:
                    # First and only step, ignore prev_a (this is a placeholder)
                    return self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1))
                else:
                    return embeddings.gather(
                        1,
                        torch.cat((first_a, prev_a), 1)[:, :, None].expand(batch_size, 2, embeddings.size(-1))
                    ).view(batch_size, 1, -1)
            # More than one step, assume always starting with first
            embeddings_per_step = embeddings.gather(
                1,
                prev_a[:, 1:, None].expand(batch_size, num_steps - 1, embeddings.size(-1))
            )
            return torch.cat((
                # First step placeholder, cat in dim 1 (time steps)
                self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1)),
                # Second step, concatenate embedding of first with embedding of current/previous (in dim 2, context dim)
                torch.cat((
                    embeddings_per_step[:, 0:1, :].expand(batch_size, num_steps - 1, embeddings.size(-1)),
                    embeddings_per_step
                ), 2)
            ), 1)

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        heads = torch.matmul(F.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = F.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2)

    def _get_mask(self, demands_with_depot, used_capacity, prev_a, mask):
        
        if self.is_vrp:
            
            if self.allow_partial:
                # Nodes that cannot be visited are already visited or the vehicle is full
                mask_loc = (demands_with_depot[:, :, 1:] == 0) | (used_capacity[:, :, None] >= self.VEHICLE_CAPACITY)
            else:
                # Nodes that cannot be visited are already visited or too much demand to be served now
                demands_loc = demands_with_depot[:, :, 1:]
                mask_loc = (
                        (demands_loc == 0) |
                        (demands_loc + used_capacity[:, :, None] > self.VEHICLE_CAPACITY)
                )
                
            # Cannot visit the depot if just visited and still unserved nodes
            mask_depot = (prev_a == 0) & ((mask_loc == 0).int().sum(-1) > 0)
            return torch.cat((mask_depot[:, :, None], mask_loc), -1)
        
        # TSP
        return mask

    def _get_attention_node_data(self, attention_node_data_fixed, demands_with_depot):

        if self.is_vrp and self.allow_partial:

            glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = attention_node_data_fixed
            # Need to provide information of how much each node has already been served
            # Clone demands as they are needed by the backprop whereas they are updated later
            glimpse_key_step, glimpse_val_step, logit_key_step = \
                self.project_node_step(demands_with_depot[:, :, :, None].clone()).chunk(3, dim=-1)

            # Projection of concatenation is equivalent to addition of projections but this is more efficient
            return (
                self._make_heads(glimpse_key_fixed + glimpse_key_step),
                self._make_heads(glimpse_val_fixed + glimpse_val_step),
                logit_key_fixed + logit_key_step,
            )

        # TSP or VRP without split delivery
        return attention_node_data_fixed

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )
