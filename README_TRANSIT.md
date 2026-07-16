# Transit Project README

**Project:** Day-Ahead Mixed-Fleet Hub-and-Spoke Transit Optimization  
**Base repository:** `attention-learn-to-route`  
**Repository source:** https://github.com/wouterkool/attention-learn-to-route

## Overview

This project extends the original `attention-learn-to-route` repository beyond the TSP benchmark
into a **day-ahead shared-transit optimization system**. The final system is designed for mixed-fleet
hub-and-spoke operations, where passenger requests are grouped into clusters, hubs are selected, a
heterogeneous fleet is assigned, and stop order is constructed and validated before the service day
begins.

The work in this repository covers two stages:

1. **Task 4: Baseline reproduction**
   - reproduce the published TSP results from the original paper
   - validate the pretrained checkpoints shipped with the repo
2. **Task 5: Novel transit model**
   - identify research gaps from experiments
   - design and validate a transit-specific routing and planning pipeline

## Problem Statement

The target problem is not a pure shortest-tour task. It is a day-ahead transit planning problem with:

- passenger demand clustering
- transfer hub selection
- mixed fleet assignment
- route ordering
- feasibility validation
- utilization-aware planning

The system must produce an executable plan before the service day starts.

## Architecture

Implemented modules:

1. **Demand clustering**
   - groups passenger requests into spatial clusters
2. **Hub selection**
   - proposes service hubs from the clustered demand
3. **Mixed-fleet assignment**
   - assigns feeder and trunk vehicles
4. **Route construction**
   - orders stops inside each route
5. **Feasibility validation**
   - checks capacity, coverage, and route consistency

Route ordering has two learned variants:

- a transit-specific learned route-ordering model
- a joint hub-and-fleet planner prototype

## Repository Structure

Key files and folders:

- `run.py` - training entry point for the original attention model
- `eval.py` - evaluation entry point for pretrained checkpoints
- `generate_data.py` - TSP dataset generation
- `transit/` - transit-specific clustering, fleet, routing, and planning modules
- `pretrained/` - shipped pretrained checkpoints
- `outputs/` - trained local checkpoints
- `END2END_REPORT.md` - submission-ready end-to-end report
- `TASK_4_5_REPORT.md` - detailed working report
- `NOVEL_MODEL_REPORT.md` - standalone novel-model report
- `TRANSIT_BENCHMARK_RESULTS.md` - benchmark summary
- `SYSTEM_DESIGN.md` - system design document

## Environment

Validated runtime:

- Python 3.11
- PyTorch 2.5.1 CPU build
- NumPy 2.4.6
- SciPy 1.17.1
- tqdm 4.68.4

Hardware used for the validated runs:

- OS: Linux 7.1.3-arch1-3 x86_64
- CPU: Intel Core Ultra 5 125H
- CPU cores: 18 logical CPUs
- RAM: 15 GiB
- CUDA: not used
- GPU: not used

Note:

- the original `environment.yml` is outdated for modern conda because it references `anaconda`
- a working local conda environment was used instead

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/wouterkool/attention-learn-to-route.git
cd attention-learn-to-route
```

### 2. Activate the working environment

If you already created the environment used in this project:

```bash
conda activate /tmp/attention_tsp
```

If you need to inspect the environment:

```bash
conda list
python -V
```

### 3. Generate the official TSP datasets

```bash
python generate_data.py --problem tsp --name test --seed 1234
python generate_data.py --problem tsp --name validation --seed 4321
```

## Baseline Reproduction

Run the pretrained checkpoints with greedy decoding:

```bash
python eval.py data/tsp/tsp20_test_seed1234.pkl --model pretrained/tsp_20 --decode_strategy greedy --no_cuda --no_progress_bar
python eval.py data/tsp/tsp50_test_seed1234.pkl --model pretrained/tsp_50 --decode_strategy greedy --no_cuda --no_progress_bar
python eval.py data/tsp/tsp100_test_seed1234.pkl --model pretrained/tsp_100 --decode_strategy greedy --no_cuda --no_progress_bar
```

### Published vs Local Results

| Problem size | Published AM greedy | Local eval |
|---|---:|---:|
| TSP20 | 3.85 | 3.8469569683 |
| TSP50 | 5.80 | 5.7903079987 |
| TSP100 | 8.12 | 8.1018714905 |

## Transit Novelty

The novel contribution is a **mixed-fleet hub-and-spoke transit pipeline**. The transit-specific
learned route-orderer is the strongest validated component. A learned joint planner for hub choice
and fleet assignment is also implemented and benchmarked.

## Training and Benchmark Commands

### Learned Route-Ordering Model

```bash
python -m transit.train_order_model --train-instances 200 --val-instances 40 --base-seed 6000 --num-hubs 3 --epochs 6 --batch-size 8 --hidden-dim 96 --lr 0.001 --save-path outputs/transit_orderer.pt
python -m transit.validate_order_model --checkpoint outputs/transit_orderer.pt --instances 30 --base-seed 9000 --num-hubs 3
python -m transit.benchmark_ordering --num-instances 20 --base-seed 1234 --num-hubs 3
python -m transit.custom_instance --output /tmp/custom_transit_instance.json --seed 2026 --num-clusters 4 --requests-per-cluster 10 --cluster-spread 0.018
python -m transit.benchmark_pipeline_metrics --instances 10 --base-seed 1234 --num-hubs 3
python -m transit.benchmark_pipeline_metrics --instance-json /tmp/custom_transit_instance.json
```

### Joint Hub-and-Fleet Planner

```bash
python -m transit.train_joint_planner --train-instances 200 --val-instances 40 --base-seed 7000 --num-hubs 3 --num-candidates 6 --epochs 8 --hidden-dim 64 --lr 0.001 --save-path outputs/transit_joint_policy.pt --learned-order-path outputs/transit_orderer.pt
python -m transit.benchmark_joint_planner --instances 20 --base-seed 1234 --num-hubs 3 --joint-policy-path outputs/transit_joint_policy.pt --learned-order-path outputs/transit_orderer.pt
python -m transit.benchmark_joint_planner --instance-json /tmp/custom_transit_instance.json --joint-policy-path outputs/transit_joint_policy.pt --learned-order-path outputs/transit_orderer.pt
```

## Validated Results

### Route Ordering

| Strategy | Mean route length | Wins vs heuristic |
|---|---:|---:|
| Heuristic | 0.867255 | 20 |
| TSP attention | 1.014257 | 0 |
| Learned transit orderer | 0.828558 | 17 |

### Full Pipeline

| Strategy | Mean route length | Mean utilization |
|---|---:|---:|
| Heuristic | 0.889041 | 0.701506 |
| TSP attention | 0.973619 | 0.701506 |
| Learned transit orderer | 0.827686 | 0.701506 |

### Joint Planner

| Strategy | Mean route length | Mean utilization |
|---|---:|---:|
| Heuristic | 0.901793 | 0.697544 |
| Learned route order only | 0.835070 | 0.697544 |
| Joint learned hub + fleet + order | 1.075017 | 0.723439 |

### Interpretation

- the pretrained TSP attention model does not transfer well to the transit setting
- the transit-specific learned route-orderer is the best route-construction component
- the joint planner is functional and demand-safe, but still needs more training to beat the route-order-only model on route length

## Research Gaps Identified

Validated gaps from the baseline experiments:

- scaling degradation as graph size increases
- sequential decoding bottleneck
- high runtime for larger instances
- size-specific checkpoints
- synthetic-only benchmark distribution
- limited generalization outside the TSP benchmark

These gaps motivate the transit-specific mixed-fleet model.


## Related Documents

- `END2END_REPORT.md`
- `TASK_4_5_REPORT.md`
- `NOVEL_MODEL_REPORT.md`
- `TRANSIT_BENCHMARK_RESULTS.md`
- `SYSTEM_DESIGN.md`

## Final Note

This README is the concise project entry point. Use `END2END_REPORT.md` as the formal submission
report and `README_TRANSIT.md` as the top-level project summary.
