from torch.utils.data import Dataset
import torch
import os
import pickle


class CVRP(object):

    NAME = 'cvrp'  # Capacitated Vehicle Routing Problem

    @staticmethod
    def get_costs(dataset, pi):
        """
        :param dataset: (batch_size, graph_size, 2) coordinates
        :param pi: (batch_size, graph_size) permutations representing tours
        :return: (batch_size) lengths of tours
        """
        batch_size, graph_size = dataset['demand'].size()
        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = pi.data.sort(1)[0]

        # Sorting it should give all zeros at front and then 1...n
        assert (
            torch.arange(1, graph_size + 1, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size) ==
            sorted_pi[:, -graph_size:]
        ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        CAPACITY = 1.
        demand_with_depot = torch.cat(
            (
                dataset['demand'][:, :1] * 0 - CAPACITY,  # Hacky
                dataset['demand']
            ),
            1
        )
        d = demand_with_depot.gather(1, pi)

        used_cap = dataset['demand'][:, 0] * 0  #  torch.zeros(batch_size, 1, out=dataset['demand'].data.new())
        for i in range(pi.size(1)):
            used_cap += d[:, i]
            # Cannot use less than 0
            used_cap[used_cap < 0] = 0
            assert (used_cap <= CAPACITY).all(), "Used more than capacity"

        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        # Length is distance (L2-norm of difference) from each next location from its prev and of first and last to depot
        return (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
            + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
            + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
        ), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPDataset(*args, **kwargs)


class SDVRP(object):

    NAME = 'sdvrp'  # Split Delivery Vehicle Routing Problem

    @staticmethod
    def get_costs(dataset, pi):
        """
        :param dataset: (batch_size, graph_size, 2) coordinates
        :param pi: (batch_size, graph_size) permutations representing tours
        :return: (batch_size) lengths of tours
        """
        batch_size, graph_size = dataset['demand'].size()

        # Each node can be visited multiple times, but we always deliver as much demand as possible
        # We check that at the end all demand has been satisfied
        CAPACITY = 1.
        demands = torch.cat(
            (
                dataset['demand'][:, :1] * 0 - CAPACITY,  # Hacky
                dataset['demand']
            ),
            1
        )
        rng = torch.arange(batch_size, out=demands.data.new().long())
        used_cap = dataset['demand'][:, 0] * 0  #  torch.zeros(batch_size, 1, out=dataset['demand'].data.new())
        a_prev = None
        for a in pi.transpose(0, 1):
            assert a_prev is None or (demands[((a_prev == 0) & (a == 0))[:, None]] == 0).all(), \
                "Cannot visit depot twice if any nonzero demand"
            d = torch.min(demands[rng, a], CAPACITY - used_cap)
            demands[rng, a] -= d
            used_cap += d
            used_cap[a == 0] = 0
            a_prev = a
        assert (demands == 0).all(), "All demand must be satisfied"

        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        # Length is distance (L2-norm of difference) from each next location from its prev and of first and last to depot
        return (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
            + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
            + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
        ), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPDataset(*args, **kwargs)


class VRPDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=1000000):
        super(VRPDataset, self).__init__()

        # From VRP with RL paper https://arxiv.org/abs/1802.04240
        CAPACITIES = {
            10: 20.,
            20: 30.,
            50: 40.,
            100: 50.
        }

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [
                    {
                        'loc': torch.FloatTensor(loc),
                        'demand': torch.FloatTensor(demand) / capacity,  # Normalize so capacity = 1
                        'depot': torch.FloatTensor(depot),
                    }
                    for depot, loc, demand, capacity in data[:num_samples]
                ]
        else:
            self.data = [
                # {
                #     'loc': torch.FloatTensor(size, 2).uniform_(0, 1),
                #     'demand': torch.FloatTensor(size).uniform_(0, 1),
                #     'depot': torch.FloatTensor(2).uniform_(0, 1)
                # }
                {
                    'loc': torch.FloatTensor(size, 2).uniform_(0, 1),
                    # Uniform 1 - 9, scaled by capacities
                    'demand': (torch.FloatTensor(size).uniform_(0, 9).int() + 1).float() / CAPACITIES[size],
                    'depot': torch.FloatTensor(2).uniform_(0, 1)
                }

                for i in range(num_samples)
            ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
