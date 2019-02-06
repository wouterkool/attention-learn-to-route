import torch
from problems.op.state_op import StateOP


def op_tsiligirides(batch, sample=False, power=4.0):
    state = StateOP.initialize(batch)

    all_a = []
    while not state.all_finished():
        # Compute scores
        mask = state.get_mask()
        p = (
                (mask[..., 1:] == 0).float() *
                state.prize[state.ids, 1:] /
                ((state.coords[state.ids, 1:, :] - state.cur_coord[:, :, None, :]).norm(p=2, dim=-1) + 1e-6)
        ) ** power
        bestp, besta = p.topk(4, dim=-1)
        bestmask = mask[..., 1:].gather(-1, besta)

        # If no feasible actions, must go to depot
        # mask == 0 means feasible, so if mask == 0 sums to 0 there are no feasible and
        # all corresponding ps should be 0, so we need to add a column with a 1 that corresponds
        # to selecting the end destination
        to_depot = ((bestmask == 0).sum(-1, keepdim=True) == 0).float()
        # best_p should be zero if we have to go to depot, but because of numeric stabilities, it isn't
        p_ = torch.cat((to_depot, bestp), -1)
        pnorm = p_ / p_.sum(-1, keepdim=True)

        if sample:
            a = pnorm[:, 0, :].multinomial(1)  # Sample action
        else:
            # greedy
            a = pnorm[:, 0, :].max(-1)[1].unsqueeze(-1)  # Add 'sampling dimension'

        # a == 0 means depot, otherwise subtract one
        final_a = torch.cat((torch.zeros_like(besta[..., 0:1]), besta + 1), -1)[:, 0, :].gather(-1, a)

        selected = final_a[..., 0]  # Squeeze unnecessary sampling dimension
        state = state.update(selected)
        all_a.append(selected)
    return torch.stack(all_a, -1)

