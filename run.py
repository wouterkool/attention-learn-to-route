#!/usr/bin/env python

import os
import json
import pprint as pp

import torch
import torch.optim as optim
from torch.nn import DataParallel
from tensorboard_logger import configure

from critic_network import CriticNetwork
from options import get_options
from train import train_epoch, validate
from baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline
from attention_model import AttentionModel
from problem_tsp import TSP as problem


def maybe_cuda_model(model, cuda, parallel=True):
    if cuda:
        model.cuda()

    if parallel and torch.cuda.device_count() > 1:
        model = DataParallel(model)

    return model

if __name__ == "__main__":
    opts = get_options()

    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)

    # Optionally configure tensorboard
    if not opts.no_tensorboard:
        configure(os.path.join(opts.log_dir, "{}_{}".format(problem.NAME, opts.graph_size), opts.run_name))

    os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    # Load data from load_path
    load_data = {}
    if opts.load_path is not None:
        print('  [*] Loading data from {}'.format(opts.load_path))
        load_data = torch.load(opts.load_path, map_location=lambda storage, loc: storage)  # Load on CPU

    # Initialize model
    model = maybe_cuda_model(
        AttentionModel(
            opts.embedding_dim,
            opts.hidden_dim,
            problem,
            n_encode_layers=opts.n_encode_layers,
            mask_inner=True,
            mask_logits=True,
            normalization=opts.normalization,
            tanh_clipping=opts.tanh_clipping
        ),
        opts.use_cuda
    )

    # Overwrite model parameters by parameters to load
    model.load_state_dict({**model.state_dict(), **load_data.get('model', {})})

    # Initialize baseline
    if opts.baseline == 'exponential':
        baseline = ExponentialBaseline(opts.exp_beta)
    elif opts.baseline == 'critic':
        baseline = CriticBaseline(
            maybe_cuda_model(
                CriticNetwork(
                    problem.NODE_DIM,
                    opts.embedding_dim,
                    opts.hidden_dim,
                    opts.n_encode_layers,
                    opts.normalization
                ),
                opts.use_cuda
            )
        )
    elif opts.baseline == 'rollout':
        baseline = RolloutBaseline(model, problem, opts)
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()

    # Load baseline from data, make sure script is called with same type of baseline
    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])

    # Initialize optimizer
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': float(opts.lr_model)}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': float(opts.lr_critic)}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    # Load optimizer state
    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

    # Start the actual training loop
    val_dataset = problem.make_dataset(size=opts.graph_size, num_samples=opts.val_size)

    if opts.eval_only:
        validate(model, val_dataset, opts)
    else:
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            train_epoch(
                model,
                optimizer,
                baseline,
                lr_scheduler,
                epoch,
                val_dataset,
                problem,
                opts
            )
