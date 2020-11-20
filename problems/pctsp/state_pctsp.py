import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter
import torch.nn.functional as F


class StatePCTSP(NamedTuple):
    # Fixed input
    coords: torch.Tensor  # Depot + loc
    expected_prize: torch.Tensor
    real_prize: torch.Tensor
    penalty: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and prizes tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    prev_a: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    cur_total_prize: torch.Tensor
    cur_total_penalty: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.coords.size(-2))

    @property
    def dist(self):
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return self._replace(
            ids=self.ids[key],
            prev_a=self.prev_a[key],
            visited_=self.visited_[key],
            lengths=self.lengths[key],
            cur_total_prize=self.cur_total_prize[key],
            cur_total_penalty=self.cur_total_penalty[key],
            cur_coord=self.cur_coord[key],
        )

    # Warning: cannot override len of NamedTuple, len should be number of fields, not batch size
    # def __len__(self):
    #     return len(self.used_capacity)

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8, stochastic=False):
        depot = input['depot']
        loc = input['loc']
        # For both deterministic and stochastic variant, model sees only deterministic (expected) prize
        expected_prize = input['deterministic_prize']
        # This is the prize that is actually obtained at each node
        real_prize = input['stochastic_prize' if stochastic else 'deterministic_prize']
        penalty = input['penalty']

        batch_size, n_loc, _ = loc.size()
        coords = torch.cat((depot[:, None, :], loc), -2)
        # For prize, prepend 0 (corresponding to depot) so we can gather efficiently

        real_prize_with_depot = torch.cat((torch.zeros_like(real_prize[:, :1]), real_prize), -1)
        penalty_with_depot = F.pad(penalty, (1, 0), mode='constant', value=0)

        return StatePCTSP(
            coords=coords,
            expected_prize=expected_prize,
            real_prize=real_prize_with_depot,
            penalty=penalty_with_depot,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently (if there is an action for depot)
                torch.zeros(
                    batch_size, 1, n_loc + 1,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_total_prize=torch.zeros(batch_size, 1, device=loc.device),
            cur_total_penalty=penalty.sum(-1)[:, None],  # Sum penalties (all when nothing is visited), add step dim
            cur_coord=input['depot'][:, None, :],  # Add step dimension
            i=torch.zeros(1, dtype=torch.int64, device=loc.device)  # Vector with length num_steps
        )

    def get_remaining_prize_to_collect(self):
        # returns the remaining prize to collect, or 0 if already collected the minimum (1.0)
        return torch.clamp(1 - self.cur_total_prize, min=0)

    def get_final_cost(self):

        assert self.all_finished()
        # assert self.visited_.
        # We are at the depot so no need to add remaining distance
        return self.lengths + self.cur_total_penalty

    def update(self, selected):

        assert self.i.size(0) == 1, "Can only update if state represents single step"

        # Update the state
        selected = selected[:, None]  # Add dimension for step
        prev_a = selected

        # Add the length
        cur_coord = self.coords[self.ids, selected]
        lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)
        # Add current total prize
        cur_total_prize = self.cur_total_prize + self.real_prize[self.ids, selected]
        cur_total_penalty = self.cur_total_penalty + self.penalty[self.ids, selected]

        if self.visited_.dtype == torch.uint8:
            # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            # This works, by check_unset=False it is allowed to set the depot visited a second a time
            visited_ = mask_long_scatter(self.visited_, prev_a, check_unset=False)

        return self._replace(
            prev_a=prev_a, visited_=visited_,
            lengths=lengths, cur_total_prize=cur_total_prize, cur_total_penalty=cur_total_penalty, cur_coord=cur_coord,
            i=self.i + 1
        )

    def all_finished(self):
        # All must be returned to depot (and at least 1 step since at start also prev_a == 0)
        # This is more efficient than checking the mask
        return self.i.item() > 0 and (self.prev_a == 0).all()
        # return self.visited[:, :, 0].all()  # If we have visited the depot we're done

    def get_current_node(self):
        """
        Returns the current node where 0 is depot, 1...n are nodes
        :return: (batch_size, num_steps) tensor with current nodes
        """
        return self.prev_a

    def get_mask(self):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited
        :return:
        """

        # Note: this always allows going to the depot, but that should always be suboptimal so be ok
        # Cannot visit if already visited or if the depot has already been visited then we cannot visit anymore
        visited_ = self.visited
        mask = (
            visited_ | visited_[:, :, 0:1]
        )
        # Cannot visit depot if not yet collected 1 total prize and there are unvisited nodes
        mask[:, :, 0] = (self.cur_total_prize < 1.) & (visited_[:, :, 1:].int().sum(-1) < visited_[:, :, 1:].size(-1))

        return mask > 0  # Hacky way to return bool or uint8 depending on pytorch version

    def construct_solutions(self, actions):
        return actions
