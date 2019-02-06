from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.pctsp.state_pctsp import StatePCTSP
from utils.beam_search import beam_search


class PCTSP(object):

    NAME = 'pctsp'  # Prize Collecting TSP, without depot, with penalties

    @staticmethod
    def _get_costs(dataset, pi, stochastic=False):
        if pi.size(-1) == 1:  # In case all tours directly return to depot, prevent further problems
            assert (pi == 0).all(), "If all length 1 tours, they should be zero"
            # Return
            return torch.zeros(pi.size(0), dtype=torch.float, device=pi.device), None

        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = pi.data.sort(1)[0]
        # Make sure each node visited once at most (except for depot)
        assert ((sorted_pi[:, 1:] == 0) | (sorted_pi[:, 1:] > sorted_pi[:, :-1])).all(), "Duplicates"

        prize = dataset['stochastic_prize'] if stochastic else dataset['deterministic_prize']
        prize_with_depot = torch.cat(
            (
                torch.zeros_like(prize[:, :1]),
                prize
            ),
            1
        )
        p = prize_with_depot.gather(1, pi)

        # Either prize constraint should be satisfied or all prizes should be visited
        assert (
            (p.sum(-1) >= 1 - 1e-5) |
            (sorted_pi.size(-1) - (sorted_pi == 0).int().sum(-1) == dataset['loc'].size(-2))
        ).all(), "Total prize does not satisfy min total prize"
        penalty_with_depot = torch.cat(
            (
                torch.zeros_like(dataset['penalty'][:, :1]),
                dataset['penalty']
            ),
            1
        )
        pen = penalty_with_depot.gather(1, pi)

        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        length = (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=-1).sum(1)  # Prevent error if len 1 seq
            + (d[:, 0] - dataset['depot']).norm(p=2, dim=-1)  # Depot to first
            + (d[:, -1] - dataset['depot']).norm(p=2, dim=-1)  # Last to depot, will be 0 if depot is last
        )
        # We want to maximize total prize but code minimizes so return negative
        # Incurred penalty cost is total penalty cost - saved penalty costs of nodes visited
        return length + dataset['penalty'].sum(-1) - pen.sum(-1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return PCTSPDataset(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        # With beam search we always consider the deterministic case
        state = PCTSPDet.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


class PCTSPDet(PCTSP):

    @staticmethod
    def get_costs(dataset, pi):
        return PCTSP._get_costs(dataset, pi, stochastic=False)

    @staticmethod
    def make_state(*args, **kwargs):
        return StatePCTSP.initialize(*args, **kwargs, stochastic=False)


class PCTSPStoch(PCTSP):

    # Stochastic variant of PCTSP, the real (stochastic) prize is only revealed when node is visited

    @staticmethod
    def get_costs(dataset, pi):
        return PCTSP._get_costs(dataset, pi, stochastic=True)

    @staticmethod
    def make_state(*args, **kwargs):
        return StatePCTSP.initialize(*args, **kwargs, stochastic=True)


def generate_instance(size, penalty_factor=3):
    depot = torch.rand(2)
    loc = torch.rand(size, 2)

    # For the penalty to make sense it should be not too large (in which case all nodes will be visited) nor too small
    # so we want the objective term to be approximately equal to the length of the tour, which we estimate with half
    # of the nodes by half of the tour length (which is very rough but similar to op)
    # This means that the sum of penalties for all nodes will be approximately equal to the tour length (on average)
    # The expected total (uniform) penalty of half of the nodes (since approx half will be visited by the constraint)
    # is (n / 2) / 2 = n / 4 so divide by this means multiply by 4 / n,
    # However instead of 4 we use penalty_factor (3 works well) so we can make them larger or smaller
    MAX_LENGTHS = {
        20: 2.,
        50: 3.,
        100: 4.
    }
    penalty_max = MAX_LENGTHS[size] * (penalty_factor) / float(size)
    penalty = torch.rand(size) * penalty_max

    # Take uniform prizes
    # Now expectation is 0.5 so expected total prize is n / 2, we want to force to visit approximately half of the nodes
    # so the constraint will be that total prize >= (n / 2) / 2 = n / 4
    # equivalently, we divide all prizes by n / 4 and the total prize should be >= 1
    deterministic_prize = torch.rand(size) * 4 / float(size)

    # In the deterministic setting, the stochastic_prize is not used and the deterministic prize is known
    # In the stochastic setting, the deterministic prize is the expected prize and is known up front but the
    # stochastic prize is only revealed once the node is visited
    # Stochastic prize is between (0, 2 * expected_prize) such that E(stochastic prize) = E(deterministic_prize)
    stochastic_prize = torch.rand(size) * deterministic_prize * 2

    return {
        'depot': depot,
        'loc': loc,
        'penalty': penalty,
        'deterministic_prize': deterministic_prize,
        'stochastic_prize': stochastic_prize
    }


class PCTSPDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(PCTSPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [
                    {
                        'depot': torch.FloatTensor(depot),
                        'loc': torch.FloatTensor(loc),
                        'penalty': torch.FloatTensor(penalty),
                        'deterministic_prize': torch.FloatTensor(deterministic_prize),
                        'stochastic_prize': torch.tensor(stochastic_prize)
                    }
                    for depot, loc, penalty, deterministic_prize, stochastic_prize in (data[offset:offset+num_samples])
                ]
        else:
            self.data = [
                generate_instance(size)
                for i in range(num_samples)
            ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
