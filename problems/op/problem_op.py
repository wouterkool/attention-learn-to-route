from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.op.state_op import StateOP
from utils.beam_search import beam_search


class OP(object):

    NAME = 'op'  # Orienteering problem

    @staticmethod
    def get_costs(dataset, pi):
        if pi.size(-1) == 1:  # In case all tours directly return to depot, prevent further problems
            assert (pi == 0).all(), "If all length 1 tours, they should be zero"
            # Return
            return torch.zeros(pi.size(0), dtype=torch.float, device=pi.device), None

        # Check that tours are valid, i.e. contain 0 to n -1
        sorted_pi = pi.data.sort(1)[0]
        # Make sure each node visited once at most (except for depot)
        assert ((sorted_pi[:, 1:] == 0) | (sorted_pi[:, 1:] > sorted_pi[:, :-1])).all(), "Duplicates"

        prize_with_depot = torch.cat(
            (
                torch.zeros_like(dataset['prize'][:, :1]),
                dataset['prize']
            ),
            1
        )
        p = prize_with_depot.gather(1, pi)

        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        length = (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=-1).sum(1)  # Prevent error if len 1 seq
            + (d[:, 0] - dataset['depot']).norm(p=2, dim=-1)  # Depot to first
            + (d[:, -1] - dataset['depot']).norm(p=2, dim=-1)  # Last to depot, will be 0 if depot is last
        )
        assert (length <= dataset['max_length'] + 1e-5).all(), \
            "Max length exceeded by {}".format((length - dataset['max_length']).max())

        # We want to maximize total prize but code minimizes so return negative
        return -p.sum(-1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return OPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateOP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = OP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


def generate_instance(size, prize_type):
    # Details see paper
    MAX_LENGTHS = {
        20: 2.,
        50: 3.,
        100: 4.
    }

    loc = torch.FloatTensor(size, 2).uniform_(0, 1)
    depot = torch.FloatTensor(2).uniform_(0, 1)
    # Methods taken from Fischetti et al. 1998
    if prize_type == 'const':
        prize = torch.ones(size)
    elif prize_type == 'unif':
        prize = (1 + torch.randint(0, 100, size=(size, ))) / 100.
    else:  # Based on distance to depot
        assert prize_type == 'dist'
        prize_ = (depot[None, :] - loc).norm(p=2, dim=-1)
        prize = (1 + (prize_ / prize_.max(dim=-1, keepdim=True)[0] * 99).int()).float() / 100.

    return {
        'loc': loc,
        # Uniform 1 - 9, scaled by capacities
        'prize': prize,
        'depot': depot,
        'max_length': torch.tensor(MAX_LENGTHS[size])
    }


class OPDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution='const'):
        super(OPDataset, self).__init__()
        assert distribution is not None, "Data distribution must be specified for OP"
        # Currently the distribution can only vary in the type of the prize
        prize_type = distribution

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [
                    {
                        'loc': torch.FloatTensor(loc),
                        'prize': torch.FloatTensor(prize),
                        'depot': torch.FloatTensor(depot),
                        'max_length': torch.tensor(max_length)
                    }
                    for depot, loc, prize, max_length in (data[offset:offset+num_samples])
                ]
        else:
            self.data = [
                generate_instance(size, prize_type)
                for i in range(num_samples)
            ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
