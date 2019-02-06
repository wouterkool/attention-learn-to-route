import time
import torch
from typing import NamedTuple
from utils.lexsort import torch_lexsort


def beam_search(*args, **kwargs):
    beams, final_state = _beam_search(*args, **kwargs)
    return get_beam_search_results(beams, final_state)


def get_beam_search_results(beams, final_state):
    beam = beams[-1]  # Final beam
    if final_state is None:
        return None, None, None, None, beam.batch_size

    # First state has no actions/parents and should be omitted when backtracking
    actions = [beam.action for beam in beams[1:]]
    parents = [beam.parent for beam in beams[1:]]

    solutions = final_state.construct_solutions(backtrack(parents, actions))
    return beam.score, solutions, final_state.get_final_cost()[:, 0], final_state.ids.view(-1), beam.batch_size


def _beam_search(state, beam_size, propose_expansions=None,
                keep_states=False):

    beam = BatchBeam.initialize(state)

    # Initial state
    beams = [beam if keep_states else beam.clear_state()]

    # Perform decoding steps
    while not beam.all_finished():

        # Use the model to propose and score expansions
        parent, action, score = beam.propose_expansions() if propose_expansions is None else propose_expansions(beam)
        if parent is None:
            return beams, None

        # Expand and update the state according to the selected actions
        beam = beam.expand(parent, action, score=score)

        # Get topk
        beam = beam.topk(beam_size)

        # Collect output of step
        beams.append(beam if keep_states else beam.clear_state())

    # Return the final state separately since beams may not keep state
    return beams, beam.state


class BatchBeam(NamedTuple):
    """
    Class that keeps track of a beam for beam search in batch mode.
    Since the beam size of different entries in the batch may vary, the tensors are not (batch_size, beam_size, ...)
    but rather (sum_i beam_size_i, ...), i.e. flattened. This makes some operations a bit cumbersome.
    """
    score: torch.Tensor  # Current heuristic score of each entry in beam (used to select most promising)
    state: None  # To track the state
    parent: torch.Tensor
    action: torch.Tensor
    batch_size: int  # Can be used for optimizations if batch_size = 1
    device: None  # Track on which device

    # Indicates for each row to which batch it belongs (0, 0, 0, 1, 1, 2, ...), managed by state
    @property
    def ids(self):
        return self.state.ids.view(-1)  # Need to flat as state has steps dimension

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):  # If tensor, idx all tensors by this tensor:
            return self._replace(
                # ids=self.ids[key],
                score=self.score[key] if self.score is not None else None,
                state=self.state[key],
                parent=self.parent[key] if self.parent is not None else None,
                action=self.action[key] if self.action is not None else None
            )
        return super(BatchBeam, self).__getitem__(key)

    # Do not use __len__ since this is used by namedtuple internally and should be number of fields
    # def __len__(self):
    #     return len(self.ids)

    @staticmethod
    def initialize(state):
        batch_size = len(state.ids)
        device = state.ids.device
        return BatchBeam(
            score=torch.zeros(batch_size, dtype=torch.float, device=device),
            state=state,
            parent=None,
            action=None,
            batch_size=batch_size,
            device=device
        )

    def propose_expansions(self):
        mask = self.state.get_mask()
        # Mask always contains a feasible action
        expansions = torch.nonzero(mask[:, 0, :] == 0)
        parent, action = torch.unbind(expansions, -1)
        return parent, action, None

    def expand(self, parent, action, score=None):
        return self._replace(
            score=score,  # The score is cleared upon expanding as it is no longer valid, or it must be provided
            state=self.state[parent].update(action),  # Pass ids since we replicated state
            parent=parent,
            action=action
        )

    def topk(self, k):
        idx_topk = segment_topk_idx(self.score, k, self.ids)
        return self[idx_topk]

    def all_finished(self):
        return self.state.all_finished()

    def cpu(self):
        return self.to(torch.device('cpu'))

    def to(self, device):
        if device == self.device:
            return self
        return self._replace(
            score=self.score.to(device) if self.score is not None else None,
            state=self.state.to(device),
            parent=self.parent.to(device) if self.parent is not None else None,
            action=self.action.to(device) if self.action is not None else None
        )

    def clear_state(self):
        return self._replace(state=None)

    def size(self):
        return self.state.ids.size(0)


def segment_topk_idx(x, k, ids):
    """
    Finds the topk per segment of data x given segment ids (0, 0, 0, 1, 1, 2, ...).
    Note that there may be fewer than k elements in a segment so the returned length index can vary.
    x[result], ids[result] gives the sorted elements per segment as well as corresponding segment ids after sorting.
    :param x:
    :param k:
    :param ids:
    :return:
    """
    assert x.dim() == 1
    assert ids.dim() == 1

    # Since we may have varying beam size per batch entry we cannot reshape to (batch_size, beam_size)
    # And use default topk along dim -1, so we have to be creative
    # Now we have to get the topk per segment which is really annoying :(
    # we use lexsort on (ids, score), create array with offset per id
    # offsets[ids] then gives offsets repeated and only keep for which arange(len) < offsets + k
    splits_ = torch.nonzero(ids[1:] - ids[:-1])

    if len(splits_) == 0:  # Only one group
        _, idx_topk = x.topk(min(k, x.size(0)))
        return idx_topk

    splits = torch.cat((ids.new_tensor([0]), splits_[:, 0] + 1))
    # Make a new array in which we store for each id the offset (start) of the group
    # This way ids does not need to be increasing or adjacent, as long as each group is a single range
    group_offsets = splits.new_zeros((splits.max() + 1,))
    group_offsets[ids[splits]] = splits
    offsets = group_offsets[ids]  # Look up offsets based on ids, effectively repeating for the repetitions per id

    # We want topk so need to sort x descending so sort -x (be careful with unsigned data type!)
    idx_sorted = torch_lexsort((-(x if x.dtype != torch.uint8 else x.int()).detach(), ids))

    # This will filter first k per group (example k = 2)
    # ids     = [0, 0, 0, 1, 1, 1, 1, 2]
    # splits  = [0, 3, 7]
    # offsets = [0, 0, 0, 3, 3, 3, 3, 7]
    # offs+2  = [2, 2, 2, 5, 5, 5, 5, 9]
    # arange  = [0, 1, 2, 3, 4, 5, 6, 7]
    # filter  = [1, 1, 0, 1, 1, 0, 0, 1]
    # Use filter to get only topk of sorting idx
    return idx_sorted[torch.arange(ids.size(0), out=ids.new()) < offsets + k]


def backtrack(parents, actions):

    # Now backtrack to find aligned action sequences in reversed order
    cur_parent = parents[-1]
    reversed_aligned_sequences = [actions[-1]]
    for parent, sequence in reversed(list(zip(parents[:-1], actions[:-1]))):
        reversed_aligned_sequences.append(sequence.gather(-1, cur_parent))
        cur_parent = parent.gather(-1, cur_parent)

    return torch.stack(list(reversed(reversed_aligned_sequences)), -1)


class CachedLookup(object):

    def __init__(self, data):
        self.orig = data
        self.key = None
        self.current = None

    def __getitem__(self, key):
        assert not isinstance(key, slice), "CachedLookup does not support slicing, " \
                                           "you can slice the result of an index operation instead"

        if torch.is_tensor(key):  # If tensor, idx all tensors by this tensor:

            if self.key is None:
                self.key = key
                self.current = self.orig[key]
            elif len(key) != len(self.key) or (key != self.key).any():
                self.key = key
                self.current = self.orig[key]

            return self.current

        return super(CachedLookup, self).__getitem__(key)
