import torch
from typing import NamedTuple


class StateSDVRP(NamedTuple):
    # Fixed input
    coords: torch.Tensor
    demand: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and demands tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    prev_a: torch.Tensor
    used_capacity: torch.Tensor
    demands_with_depot: torch.Tensor  # Keeps track of remaining demands
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step

    VEHICLE_CAPACITY = 1.0  # Hardcoded

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return self._replace(
            ids=self.ids[key],
            prev_a=self.prev_a[key],
            used_capacity=self.used_capacity[key],
            demands_with_depot=self.demands_with_depot[key],
            lengths=self.lengths[key],
            cur_coord=self.cur_coord[key],
        )

    @staticmethod
    def initialize(input):

        depot = input['depot']
        loc = input['loc']
        demand = input['demand']

        batch_size, n_loc, _ = loc.size()
        return StateSDVRP(
            coords=torch.cat((depot[:, None, :], loc), -2),
            demand=demand,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            used_capacity=demand.new_zeros(batch_size, 1),
            demands_with_depot=torch.cat((
                demand.new_zeros(batch_size, 1),
                demand[:, :]
            ), 1)[:, None, :],
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=input['depot'][:, None, :],  # Add step dimension
            i=torch.zeros(1, dtype=torch.int64, device=loc.device)  # Vector with length num_steps
        )

    def get_final_cost(self):

        assert self.all_finished()

        return self.lengths + (self.coords[self.ids, 0, :] - self.cur_coord).norm(p=2, dim=-1)

    def update(self, selected):

        assert self.i.size(0) == 1, "Can only update if state represents single step"

        # Update the state
        selected = selected[:, None]  # Add dimension for step
        prev_a = selected

        # Add the length
        cur_coord = self.coords[self.ids, selected]
        lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)

        # Not selected_demand is demand of first node (by clamp) so incorrect for nodes that visit depot!
        selected_demand = self.demands_with_depot.gather(-1, prev_a[:, :, None])[:, :, 0]
        delivered_demand = torch.min(selected_demand, self.VEHICLE_CAPACITY - self.used_capacity)

        # Increase capacity if depot is not visited, otherwise set to 0
        #used_capacity = torch.where(selected == 0, 0, self.used_capacity + delivered_demand)
        used_capacity = (self.used_capacity + delivered_demand) * (prev_a != 0).float()

        # demands_with_depot = demands_with_depot.clone()[:, 0, :]
        # Add one dimension since we write a single value
        demands_with_depot = self.demands_with_depot.scatter(
            -1,
            prev_a[:, :, None],
            self.demands_with_depot.gather(-1, prev_a[:, :, None]) - delivered_demand[:, :, None]
        )
        
        return self._replace(
            prev_a=prev_a, used_capacity=used_capacity, demands_with_depot=demands_with_depot,
            lengths=lengths, cur_coord=cur_coord, i=self.i + 1
        )

    def all_finished(self):
        return self.i.item() >= self.demands_with_depot.size(-1) and not (self.demands_with_depot > 0).any()

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited
        :return:
        """

        # Nodes that cannot be visited are already visited or too much demand to be served now
        mask_loc = (self.demands_with_depot[:, :, 1:] == 0) | (self.used_capacity[:, :, None] >= self.VEHICLE_CAPACITY)

        # Cannot visit the depot if just visited and still unserved nodes
        mask_depot = (self.prev_a == 0) & ((mask_loc == 0).int().sum(-1) > 0)
        return torch.cat((mask_depot[:, :, None], mask_loc), -1)

    def construct_solutions(self, actions):
        return actions
