# End-to-End Submission Report: Baseline Reproduction, Research Gaps, and Transit Novelty

**Project:** `attention-learn-to-route` baseline reproduction and mixed-fleet transit novelty  
**Student:** `[Your Name]`  

## Abstract

This report first reproduces the
published baseline results from the `attention-learn-to-route` repository on TSP20, TSP50, and
TSP100. It then uses the reproduced baseline to identify validated research gaps and motivates a
novel **day-ahead mixed-fleet hub-and-spoke transit optimization** framework.

The final validated system is not a plain TSP solver. It is a transit planning pipeline with:

- demand clustering
- hub selection
- mixed-fleet assignment
- learned route ordering
- a learned joint-planner prototype for hub and fleet decisions

The strongest validated component is the transit-specific learned route-ordering module. The joint
hub-and-fleet planner is functional and demand-safe

Reference motivation:

- [Flexible Network Strategy Mixed Fleet Hub-and-Spoke Transit Optimization.pdf](/home/ygn/Documents/RV Univ/Simulation/Flexible%20Network%20Strategy%20Mixed%20Fleet%20Hub-and-Spoke%20Transit%20Optimization.pdf)

Repository source:

- [https://github.com/wouterkool/attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route)

---

## 1. Task 4: Baseline Reproduction

### 1.1 Environment and Setup

The repository was run in a working local conda environment 

The validated runtime stack used here is:

- Python 3.11
- PyTorch CPU build
- NumPy
- SciPy
- tqdm

Hardware used for the runs:

- OS: Linux 7.1.3-arch1-3 x86_64
- CPU: Intel Core Ultra 5 125H
- CPU cores: 18 logical CPUs
- RAM: 15 GiB
- CUDA: not used
- GPU: not used

The baseline evaluation used the pretrained checkpoints included in the repo and the official test
dataset generator.

### 1.2 Commands Used

```bash
conda activate /tmp/attention_tsp

python generate_data.py --problem tsp --name test --seed 1234
python generate_data.py --problem tsp --name validation --seed 4321

python eval.py data/tsp/tsp20_test_seed1234.pkl --model pretrained/tsp_20 --decode_strategy greedy --no_cuda --no_progress_bar
python eval.py data/tsp/tsp50_test_seed1234.pkl --model pretrained/tsp_50 --decode_strategy greedy --no_cuda --no_progress_bar
python eval.py data/tsp/tsp100_test_seed1234.pkl --model pretrained/tsp_100 --decode_strategy greedy --no_cuda --no_progress_bar
```

### 1.3 Reproduction Results

| Problem size | Published AM greedy | Local eval | Difference |
|---|---:|---:|---:|
| TSP20 | 3.85 | 3.8469569683 | -0.0030430317 |
| TSP50 | 5.80 | 5.7903079987 | -0.0096920013 |
| TSP100 | 8.12 | 8.1018714905 | -0.0181285095 |

### 1.4 Runtime Observation

| Problem size | Serial duration | Parallel duration |
|---|---:|---:|
| TSP20 | 0.2851 s | 0.000278 s |
| TSP50 | 1.7575 s | 0.001716 s |
| TSP100 | 4.8429 s | 0.004729 s |

### 1.5 Task 4 Conclusion

The baseline is reproduced successfully. The local results closely match the paper, and the same
qualitative trend is preserved: quality degrades as graph size grows and runtime increases with
instance size.

### 1.6 Installation Procedure

1. Clone the repository from the source link above.
2. Activate the working conda environment:

```bash
conda activate /tmp/attention_tsp
```

3. Verify the repository files and pretrained checkpoints are present.
4. Generate the TSP test datasets.
5. Run the evaluation commands listed above.



---

## 2. Task 5: Validated Research Gaps

The experiments justify the following research gaps.

### 2.1 Current Model Limitations

- Performance degrades as graph size increases.
- Greedy decoding is efficient but not the best-quality decoding mode.
- The baseline is size-specific rather than size-agnostic.

### 2.2 Performance Bottlenecks

- Tour construction is sequential.
- Attention cost grows with the number of nodes.
- Sampling-based decoding is expensive.

### 2.3 Computational Complexity

- Decode time is not constant.
- Wall-clock runtime rises sharply from TSP20 to TSP100.
- The architecture is suitable for benchmarking but increasingly costly for larger graphs.

### 2.4 Dataset Challenges

- The benchmark uses synthetic Euclidean instances only.
- The distribution is uniform and narrow.
- There is no map structure, traffic variability, or real-world heterogeneity.

### 2.5 Generalization Issues

- Separate checkpoints are used for different graph sizes.
- The baseline does not demonstrate robust transfer to shifted distributions.
- The benchmark optimizes only tour length, not the richer operational constraints of real routing.

### 2.6 Validated Research Gap List

| Gap | Evidence | Implication |
|---|---|---|
| Scaling degradation | TSP20 < TSP50 < TSP100 | Larger graphs are harder |
| Runtime growth | 0.2851 s -> 4.8429 s | Decode cost becomes a bottleneck |
| Size-specific behavior | Separate pretrained models | Weak transfer across graph sizes |
| Synthetic-only benchmark | Uniform Euclidean instances | Limited real-world relevance |
| Single-objective evaluation | Tour length only | Missing operational constraints |

---

## 3. Novel Model Design

The project novelty is the transit system, not the TSP benchmark.

### 3.1 Problem Statement

The target system is a **day-ahead mixed-fleet hub-and-spoke transit optimizer**. It must plan a
service day in advance by combining demand clustering, hub selection, mixed-fleet assignment, route
ordering, and feasibility validation.

### 3.2 Constraints Used in the Design

- Planning is day-ahead only.
- The fleet is heterogeneous.
- Passengers must be aggregated through hubs.
- Capacity and transfer feasibility must hold.
- The published schedule must be executable.
- The TSP attention model alone is insufficient for the full system.

### 3.3 Architecture

Implemented modules:

1. **Demand clustering**
   - groups passenger requests into spatial clusters
2. **Hub proposal / selection**
   - chooses service hubs from the demand structure
3. **Mixed-fleet assignment**
   - assigns feeder and trunk vehicles
4. **Route construction**
   - orders stops inside each route
5. **Feasibility validation**
   - checks capacity and coverage

### 3.4 What Was Learned

- The pretrained TSP attention model works as a fallback lower-level route-ordering decoder.
- A transit-specific learned route-ordering model outperforms both the heuristic and the transferred
  TSP attention decoder.
- A learned joint planner for hub choice and fleet assignment is functional and demand-safe, but it
  is not yet better than the route-order-only model on route length.

---

## 4. Validation and Benchmarks

### 4.1 Route-Ordering Benchmark

On 20 synthetic transit benchmark instances:

| Strategy | Mean route length | Wins vs heuristic |
|---|---:|---:|
| Heuristic | 0.867255 | 20 |
| TSP attention | 1.014257 | 0 |
| Learned transit orderer | 0.828558 | 17 |

### 4.2 Full Pipeline Benchmark

| Strategy | Mean route length | Mean utilization |
|---|---:|---:|
| Heuristic | 0.889041 | 0.701506 |
| TSP attention | 0.973619 | 0.701506 |
| Learned transit orderer | 0.827686 | 0.701506 |

### 4.3 Custom Instance Benchmark

| Strategy | Total route length | Utilization |
|---|---:|---:|
| Heuristic | 1.024704 | 0.750000 |
| TSP attention | 1.021507 | 0.750000 |
| Learned transit orderer | 0.913460 | 0.750000 |

### 4.4 Joint Planner Benchmark

The learned hub-selection and fleet-assignment modules were trained and benchmarked as a joint
planner.

Batch benchmark:

| Strategy | Mean route length | Mean utilization |
|---|---:|---:|
| Heuristic | 0.901793 | 0.697544 |
| Learned route order only | 0.835070 | 0.697544 |
| Joint learned hub + fleet + order | 1.075017 | 0.723439 |

Custom instance benchmark:

| Strategy | Total route length | Utilization |
|---|---:|---:|
| Heuristic | 1.024704 | 0.750000 |
| Learned route order only | 0.913460 | 0.750000 |
| Joint learned hub + fleet + order | 0.925462 | 0.750000 |

Interpretation:

- The joint planner is functional and preserves demand coverage.
- It improves utilization in the batch benchmark.
- It is close to the learned-order baseline on the custom instance.
- It is not yet better than the learned-order-only model on route length.

---

## 5. Final Outcome

The final validated outcome of the project is:

1. Baseline TSP reproduction close to the published paper.
2. A validated list of research gaps based on scaling, runtime, and generalization limits.
3. A transit-specific learned route-ordering model that beats the heuristic and the transferred TSP
   decoder.
4. A learned hub-and-fleet joint-planner prototype that is functional but still needs more training.
5. A full day-ahead mixed-fleet hub-and-spoke transit pipeline.

### Final Assessment

- The baseline attention model is reproduced successfully.
- The main research gap is not reproduction failure, but lack of scalability and weak transfer.
- The most convincing novelty is the transit-specific route-ordering module embedded in a larger
  transit system.
- The learned joint planner is a valid next step, but it is not yet the strongest performing module.

---

## 7. Submission Checklist

This section collects the items that are usually expected in a final submission package.

### 7.1 What Is Already Covered

- Source repository link
- Environment overview
- Installation / execution commands
- Training commands
- Baseline reproduction results
- Comparison against published values
- Analysis of performance differences
- Research gaps
- Novel transit model design
- Final benchmark outcomes

### 7.2 What You Should Capture Manually

Take screenshots of the terminal output for these commands:

1. Baseline evaluation:

```bash
conda activate /tmp/attention_tsp
python eval.py data/tsp/tsp20_test_seed1234.pkl --model pretrained/tsp_20 --decode_strategy greedy --no_cuda --no_progress_bar
python eval.py data/tsp/tsp50_test_seed1234.pkl --model pretrained/tsp_50 --decode_strategy greedy --no_cuda --no_progress_bar
python eval.py data/tsp/tsp100_test_seed1234.pkl --model pretrained/tsp_100 --decode_strategy greedy --no_cuda --no_progress_bar
```

2. Learned route-order training:

```bash
python -m transit.train_order_model --train-instances 200 --val-instances 40 --base-seed 6000 --num-hubs 3 --epochs 6 --batch-size 8 --hidden-dim 96 --lr 0.001 --save-path outputs/transit_orderer.pt
```

3. Learned route-order benchmark:

```bash
python -m transit.benchmark_ordering --num-instances 20 --base-seed 1234 --num-hubs 3
```

4. Joint planner training:

```bash
python -m transit.train_joint_planner --train-instances 200 --val-instances 40 --base-seed 7000 --num-hubs 3 --num-candidates 6 --epochs 8 --hidden-dim 64 --lr 0.001 --save-path outputs/transit_joint_policy.pt --learned-order-path outputs/transit_orderer.pt
```

5. Joint planner benchmarks:

```bash
python -m transit.benchmark_joint_planner --instances 20 --base-seed 1234 --num-hubs 3 --joint-policy-path outputs/transit_joint_policy.pt --learned-order-path outputs/transit_orderer.pt
python -m transit.custom_instance --output /tmp/custom_transit_instance.json --seed 2026 --num-clusters 4 --requests-per-cluster 10 --cluster-spread 0.018
python -m transit.benchmark_joint_planner --instance-json /tmp/custom_transit_instance.json --joint-policy-path outputs/transit_joint_policy.pt --learned-order-path outputs/transit_orderer.pt
```

### 7.3 What to Put in the Screenshots

Each screenshot should clearly show:

- the command being run
- the final summary metrics
- successful completion without traceback errors

For the benchmark screenshots, make sure the output includes:

- the benchmark heading
- strategy names
- mean route length / utilization numbers
- the final interpretation lines if available


---

## 8. Commands Appendix

Use the following commands in order when reproducing the full study.

### 8.1 Baseline Reproduction

```bash
conda activate /tmp/attention_tsp

python generate_data.py --problem tsp --name test --seed 1234
python generate_data.py --problem tsp --name validation --seed 4321

python eval.py data/tsp/tsp20_test_seed1234.pkl --model pretrained/tsp_20 --decode_strategy greedy --no_cuda --no_progress_bar
python eval.py data/tsp/tsp50_test_seed1234.pkl --model pretrained/tsp_50 --decode_strategy greedy --no_cuda --no_progress_bar
python eval.py data/tsp/tsp100_test_seed1234.pkl --model pretrained/tsp_100 --decode_strategy greedy --no_cuda --no_progress_bar
```

### 8.2 Learned Route-Ordering Model

```bash
python -m transit.train_order_model --train-instances 200 --val-instances 40 --base-seed 6000 --num-hubs 3 --epochs 6 --batch-size 8 --hidden-dim 96 --lr 0.001 --save-path outputs/transit_orderer.pt
python -m transit.validate_order_model --checkpoint outputs/transit_orderer.pt --instances 30 --base-seed 9000 --num-hubs 3
python -m transit.custom_instance --output /tmp/custom_transit_instance.json --seed 2026 --num-clusters 4 --requests-per-cluster 10 --cluster-spread 0.018
python -m transit.benchmark_ordering --num-instances 20 --base-seed 1234 --num-hubs 3
python -m transit.benchmark_pipeline_metrics --instances 10 --base-seed 1234 --num-hubs 3
python -m transit.benchmark_pipeline_metrics --instance-json /tmp/custom_transit_instance.json
```

### 8.3 Joint Planner

```bash
python -m transit.train_joint_planner --train-instances 200 --val-instances 40 --base-seed 7000 --num-hubs 3 --num-candidates 6 --epochs 8 --hidden-dim 64 --lr 0.001 --save-path outputs/transit_joint_policy.pt --learned-order-path outputs/transit_orderer.pt
python -m transit.benchmark_joint_planner --instances 20 --base-seed 1234 --num-hubs 3 --joint-policy-path outputs/transit_joint_policy.pt --learned-order-path outputs/transit_orderer.pt
python -m transit.benchmark_joint_planner --instance-json /tmp/custom_transit_instance.json --joint-policy-path outputs/transit_joint_policy.pt --learned-order-path outputs/transit_orderer.pt
```

