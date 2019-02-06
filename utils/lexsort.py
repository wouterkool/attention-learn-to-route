import torch
import numpy as np


def torch_lexsort(keys, dim=-1):
    if keys[0].is_cuda:
        return _torch_lexsort_cuda(keys, dim)
    else:
        # Use numpy lex sort
        return torch.from_numpy(np.lexsort([k.numpy() for k in keys], axis=dim))


def _torch_lexsort_cuda(keys, dim=-1):
    """
    Function calculates a lexicographical sort order on GPU, similar to np.lexsort
    Relies heavily on undocumented behavior of torch.sort, namely that when sorting more than
    2048 entries in the sorting dim, it performs a sort using Thrust and it uses a stable sort
    https://github.com/pytorch/pytorch/blob/695fd981924bd805704ecb5ccd67de17c56d7308/aten/src/THC/generic/THCTensorSort.cu#L330
    """

    MIN_NUMEL_STABLE_SORT = 2049  # Minimum number of elements for stable sort

    # Swap axis such that sort dim is last and reshape all other dims to a single (batch) dimension
    reordered_keys = tuple(key.transpose(dim, -1).contiguous() for key in keys)
    flat_keys = tuple(key.view(-1) for key in keys)
    d = keys[0].size(dim)  # Sort dimension size
    numel = flat_keys[0].numel()
    batch_size = numel // d
    batch_key = torch.arange(batch_size, dtype=torch.int64, device=keys[0].device)[:, None].repeat(1, d).view(-1)

    flat_keys = flat_keys + (batch_key,)

    # We rely on undocumented behavior that the sort is stable provided that
    if numel < MIN_NUMEL_STABLE_SORT:
        n_rep = (MIN_NUMEL_STABLE_SORT + numel - 1) // numel  # Ceil
        rep_key = torch.arange(n_rep, dtype=torch.int64, device=keys[0].device)[:, None].repeat(1, numel).view(-1)
        flat_keys = tuple(k.repeat(n_rep) for k in flat_keys) + (rep_key,)

    idx = None  # Identity sorting initially
    for k in flat_keys:
        if idx is None:
            _, idx = k.sort(-1)
        else:
            # Order data according to idx and then apply
            # found ordering to current idx (so permutation of permutation)
            # such that we can order the next key according to the current sorting order
            _, idx_ = k[idx].sort(-1)
            idx = idx[idx_]

    # In the end gather only numel and strip of extra sort key
    if numel < MIN_NUMEL_STABLE_SORT:
        idx = idx[:numel]

    # Get only numel (if we have replicated), swap axis back and shape results
    return idx[:numel].view(*reordered_keys[0].size()).transpose(dim, -1) % d
