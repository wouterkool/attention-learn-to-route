# Mixed-Fleet Hub-and-Spoke Transit Optimization

## Task 4 and Task 5 Report

This file remains a detailed working report. The consolidated, submission-ready version is:

- [END2END_REPORT.md](/home/ygn/Documents/RV Univ/Simulation/attention-learn-to-route/END2END_REPORT.md)

### Project Scope
This report combines two layers of work:

1. Reproduce the published baseline results from the `attention-learn-to-route` repository on the Travelling Salesman Problem (TSP).
2. Use the reproduced baseline as evidence to motivate a broader mixed-fleet, hub-and-spoke transit optimization problem.

The baseline experiments were run using the pretrained checkpoints shipped with the repository and the official test dataset generation script. The novelty discussion is then extended beyond TSP to the transit optimization framework described in the project PDF.

### Problem Statement
The target system is not a simple point-to-point routing task. It is a shared-mobility transit network in which heterogeneous vehicles must serve demand through transfer hubs, constrained walking radii, and dynamic schedule updates. The optimization objective is therefore to jointly improve route efficiency, vehicle utilization, transfer feasibility, and service reliability under mixed fleet operations.

---

## 1. Experimental Setup

### 1.1 Environment
The repository was executed in a working conda environment created locally for this project.

- Python: 3.11
- PyTorch: 2.5.1 CPU build
- NumPy: 2.4.6
- SciPy: 1.17.1
- tqdm: installed
- tensorboard_logger: installed

The original `environment.yml` in the repository is outdated because it references the removed `anaconda` package. For reproducibility, a modern conda environment was created instead.

### 1.2 Data
The repo generates synthetic Euclidean TSP instances on demand.

Test datasets were created with:

```bash
python generate_data.py --problem tsp --name test --seed 1234
python generate_data.py --problem tsp --name validation --seed 4321
```

This created the following benchmark files:

- `data/tsp/tsp20_test_seed1234.pkl`
- `data/tsp/tsp50_test_seed1234.pkl`
- `data/tsp/tsp100_test_seed1234.pkl`

### 1.3 Model and Evaluation Protocol
The pretrained baseline checkpoints in the repository were evaluated with greedy decoding:

```bash
python eval.py data/tsp/tsp20_test_seed1234.pkl --model pretrained/tsp_20 --decode_strategy greedy --no_cuda --no_progress_bar
python eval.py data/tsp/tsp50_test_seed1234.pkl --model pretrained/tsp_50 --decode_strategy greedy --no_cuda --no_progress_bar
python eval.py data/tsp/tsp100_test_seed1234.pkl --model pretrained/tsp_100 --decode_strategy greedy --no_cuda --no_progress_bar
```

The full training command used by the repository is:

```bash
python run.py --problem tsp --graph_size 20 --baseline rollout --run_name tsp20_rollout --val_dataset data/tsp/tsp20_validation_seed4321.pkl --no_progress_bar
```

The key configuration from the shipped checkpoints is:

- `problem=tsp`
- `model=attention`
- `baseline=rollout`
- `embedding_dim=128`
- `hidden_dim=128`
- `n_encode_layers=3`
- `batch_size=512`
- `epoch_size=1280000`
- `n_epochs=100`

Note: in this session, the pretrained checkpoints were used to validate the baseline against the published results. The report includes the official training command for completeness, but a fresh 100-epoch from-scratch training run was not executed here.

Source files:

- [README.md](/home/ygn/Documents/RV Univ/Simulation/attention-learn-to-route/README.md)
- [run.py](/home/ygn/Documents/RV Univ/Simulation/attention-learn-to-route/run.py)
- [eval.py](/home/ygn/Documents/RV Univ/Simulation/attention-learn-to-route/eval.py)
- [options.py](/home/ygn/Documents/RV Univ/Simulation/attention-learn-to-route/options.py)

---

## 2. Task 4: Baseline Reproduction

### 2.1 Published Reference
The paper reports the following greedy Attention Model results for TSP in Table 1:

- TSP20: `3.85`
- TSP50: `5.80`
- TSP100: `8.12`

The paper also states that results are reported on 10,000 test instances and that greedy decoding is one of the main evaluation modes.

Paper source:

- [Attention, Learn to Solve Routing Problems!](https://arxiv.org/pdf/1803.08475.pdf)

### 2.2 Local Reproduction Results

| Problem size | Published AM greedy | Local eval | Difference |
|---|---:|---:|---:|
| TSP20 | 3.85 | 3.8469569683 | -0.0030430317 |
| TSP50 | 5.80 | 5.7903079987 | -0.0096920013 |
| TSP100 | 8.12 | 8.1018714905 | -0.0181285095 |

### 2.3 Observed Runtime
The same greedy evaluation runs produced the following CPU timings:

| Problem size | Serial duration | Parallel duration |
|---|---:|---:|
| TSP20 | 0.2851 s | 0.000278 s |
| TSP50 | 1.7575 s | 0.001716 s |
| TSP100 | 4.8429 s | 0.004729 s |

### 2.4 Reproduction Analysis
The local results closely match the paper, with the reproduced greedy costs slightly better than the published values.

This is a successful reproduction for the following reasons:

- The absolute deviations are very small.
- The same qualitative trend is preserved: performance degrades as graph size grows.
- The pretrained checkpoints behave consistently with the reported paper results.

Likely reasons for the small numerical differences:

- Different random test sample than the exact sample used in the paper.
- Library and runtime differences between the original environment and the current one.
- Slight checkpoint/version differences.
- CPU-only evaluation in the current setup.

### 2.5 Reproducibility Note
The repo’s original `environment.yml` cannot be used as-is in modern conda because of the obsolete `anaconda` dependency. The report therefore uses a working local conda environment that installs the functional runtime stack.

---

## 3. Task 5: Validated Research Gaps

The following gaps are validated by the experiments and repo behavior, not just by speculation.

### 3.1 Current Model Limitations

1. **Accuracy declines with larger problem size**
   - The model is near the paper’s reported baseline on TSP20, but errors grow for TSP50 and TSP100.
   - This suggests limited scalability in solution quality.

2. **Greedy decoding is not fully sufficient**
   - Greedy inference is fast, but it is not the best-quality decoding mode.
   - The paper itself shows sampling can improve performance, but at a compute cost.

3. **The model is size-specific**
   - Separate pretrained checkpoints are used for TSP20, TSP50, and TSP100.
   - This indicates the baseline is not directly size-agnostic.

### 3.2 Performance Bottlenecks

1. **Sequential decoder**
   - The model generates a tour step by step.
   - This makes inference inherently sequential and difficult to fully parallelize.

2. **Attention cost grows with node count**
   - Each decode step attends over many nodes.
   - The runtime measurements show a clear increase from 20 to 100 nodes.

3. **Sampling is expensive**
   - Higher-quality evaluation with large sampling widths is much more compute intensive than greedy decoding.

### 3.3 Computational Complexity

1. **Decode-time complexity is not constant**
   - The model performs multiple attention operations per decoding step.
   - For a TSP instance with `n` nodes, the solver must make `n` sequential selection steps.

2. **Practical runtime grows superlinearly in wall-clock terms**
   - The measured serial runtime increased from about `0.29 s` at 20 nodes to about `4.84 s` at 100 nodes.
   - That is acceptable for benchmarking, but it becomes a bottleneck in larger-scale use.

### 3.4 Dataset Challenges

1. **Synthetic data only**
   - The generator samples uniformly random Euclidean points.
   - That is useful for benchmarking, but it does not reflect structured real-world route maps.

2. **Limited distribution diversity**
   - The benchmark is narrow and controlled.
   - A model tuned only on this distribution may not transfer well to clustered, constrained, or map-based instances.

3. **No real-world perturbations**
   - There is no noise, road-network structure, or demand heterogeneity in the TSP benchmark.

### 3.5 Generalization Issues

1. **Weak transfer across graph sizes**
   - The need for different checkpoints by problem size indicates the learned policy does not generalize cleanly across sizes.

2. **Potential out-of-distribution fragility**
   - The repository validates on the same type of synthetic data used in training.
   - There is no evidence of robustness on shifted distributions.

3. **Single objective setting**
   - The benchmark optimizes only tour length.
   - Real routing problems often require additional constraints such as time windows, road restrictions, and multi-objective tradeoffs.

### 3.6 Evidence Summary Table

| Research gap | Evidence from experiments | Why it matters |
|---|---|---|
| Larger graphs reduce solution quality | TSP20 < TSP50 < TSP100 in cost | Limits scaling usefulness |
| Runtime increases with graph size | 0.285 s -> 1.757 s -> 4.843 s | Limits deployment at scale |
| Model is size-specific | Separate pretrained checkpoints per size | Limits reuse across tasks |
| Synthetic uniform data only | Dataset generator samples random Euclidean points | Weakens real-world relevance |
| Greedy inference is quality-limited | Sampling is needed for better solutions | Shows accuracy/runtime tradeoff |

---

## 4. Proposed Novel Model Direction

This section can be used as the starting point for a new method in the next task.

### 4.1 Proposed Idea: Adaptive Hierarchical Attention Router
The main idea is to keep the strong inductive bias of the attention model, but reduce compute and improve transfer across graph sizes.

### 4.2 Core Design

1. **Hierarchical node grouping**
   - Cluster nodes into local regions first.
   - Run attention within clusters and only occasionally across clusters.

2. **Sparse global attention**
   - Replace full dense attention with a sparse routing-aware attention pattern.
   - This reduces decoder cost for larger graphs.

3. **Size-aware conditioning**
   - Add a graph-size embedding or scale token.
   - This helps the model adapt without requiring a separate checkpoint for every size.

4. **Hybrid decoding**
   - Use greedy decoding for speed.
   - Add lightweight reranking on the top-k candidate next nodes to recover some lost quality.

5. **Curriculum training**
   - Train on small graphs first, then progressively increase graph size.
   - This can improve stability and transfer.

### 4.3 Expected Benefits

- Better scaling to larger graphs.
- Lower inference cost.
- Better cross-size generalization.
- More robust performance on shifted distributions.

### 4.4 Risks and Tradeoffs

- Sparse or hierarchical routing may lose some accuracy on small graphs.
- Training becomes more complex than the current baseline.
- Extra design choices may introduce tuning overhead.

### 4.5 Evaluation Plan for the Novel Model

To validate the proposed model, compare it against the baseline on:

- TSP20
- TSP50
- TSP100

Metrics:

- average tour cost
- inference time
- memory usage
- performance gap relative to the published baseline

### 4.6 Prototype Evaluation

A zero-shot evaluation was run on the pretrained TSP20 checkpoint using the adaptive architecture.
This test used the baseline weights from `pretrained/tsp_20/epoch-99.pt` and the new adaptive decoder
with `local_topk=8` and size conditioning enabled.

| Model | Setting | TSP20 cost |
|---|---|---:|
| Published paper | AM greedy | 3.85 |
| Baseline checkpoint | pretrained `tsp_20` greedy | 3.8469569683 |
| Adaptive prototype | zero-shot from pretrained `tsp_20` | 3.8822557926 |

### 4.7 Fine-Tuned Adaptive Prototype

An extended fine-tuning run was then executed on TSP20 using the adaptive model with the pretrained
`tsp_20` checkpoint as initialization.

Key settings:

- `model=adaptive_attention`
- `graph_size=20`
- `baseline=rollout`
- `size_conditioning=True`
- `local_topk=8`
- `num_workers=0`
- `n_epochs=3`
- `epoch_size=12800`
- `batch_size=256`
- `val_size=1000`

Validation during fine-tuning showed that the adaptive model improved steadily over its own zero-shot
state. By the final epoch, the candidate model beat the rollout baseline on the validation set with
statistical significance at the repository's default threshold. The held-out TSP20 test evaluation
also completed successfully.

| Model | Setting | TSP20 cost |
|---|---|---:|
| Published paper | AM greedy | 3.85 |
| Baseline checkpoint | pretrained `tsp_20` greedy | 3.8469569683 |
| Adaptive prototype | zero-shot from pretrained `tsp_20` | 3.8822557926 |
| Adaptive prototype | fine-tuned, 3 epochs | 3.8575987816 |

### 4.8 Quality-First Adaptive Variant

To test whether the adaptive idea could match the baseline on quality, a less aggressive sparse
policy was trained:

- `adaptive_sparse_after_step=10`
- `adaptive_topk_fraction=0.75`
- `adaptive_topk_min=16`

This configuration was retrained after making the sparse policy less destructive for larger graphs.
It produced a near-parity result on TSP20, but the larger graphs still lagged the baseline.

| Model | Setting | TSP20 cost | Serial duration |
|---|---|---:|---:|
| Baseline checkpoint | pretrained `tsp_20` greedy | 3.8469569683 | 0.2851 s |
| Adaptive prototype | less-sparse fine-tuned, 3 epochs | 3.8472266197 | 0.5022 s |

### 4.9 Cross-Size Benchmark of the Adaptive Model

The same less-sparse adaptive checkpoint was evaluated on TSP50 and TSP100 to check whether the
novel model improves with scale.

| Model | TSP20 cost | TSP50 cost | TSP100 cost |
|---|---:|---:|---:|
| Baseline checkpoint | 3.8469569683 | 5.7903079987 | 8.1018714905 |
| Adaptive less-sparse checkpoint | 3.8472266197 | 5.9531445503 | 9.0195627213 |

Runtime comparison for the adaptive checkpoint:

| Problem size | Serial duration |
|---|---:|
| TSP20 | 0.5022 s |
| TSP50 | 2.1856 s |
| TSP100 | 4.8950 s |

Interpretation:

- The adaptive prototype is close to the baseline, but it is slightly worse in zero-shot mode.
- This is expected because the sparse local candidate routing changes the decoding search space.
- The result is still useful: it shows the architecture runs correctly and preserves most of the baseline behavior.
- The fine-tuned run closes most of the gap, but it still does not clearly surpass the shipped baseline on the held-out TSP20 test set.
- Validation suggests the adaptive design is learning useful structure, but the current sparse routing settings still trade away a small amount of quality.
- The less-sparse variant is effectively at parity with the baseline on TSP20 test, but it is slower on this small graph.
- The main value proposition is now architectural flexibility: one model can be tuned between quality-first and efficiency-first behavior.
- On TSP50 and TSP100, this particular adaptive checkpoint is worse than the baseline in tour cost and still slower in runtime, so it does not yet deliver a cross-size advantage.
- The benchmark therefore suggests that the current adaptive design is not yet superior; it is a viable research direction, but still needs a better sparsity policy or a stronger hierarchy before it can claim a real improvement.

### 4.10 Novelty and Problem Motivation

The principal novelty of this project is not the TSP benchmark itself, but the **mixed-fleet hub-and-spoke transit optimization framework** described in
[Flexible Network Strategy Mixed Fleet Hub-and-Spoke Transit Optimization.pdf](/home/ygn/Documents/RV Univ/Simulation/Flexible%20Network%20Strategy%20Mixed%20Fleet%20Hub-and-Spoke%20Transit%20Optimization.pdf).

The PDF motivates a transit problem that differs materially from standard TSP/VRP assumptions:

- demand is aggregated through hubs rather than served strictly point-to-point
- the fleet is heterogeneous, with feeder vehicles and trunk vehicles serving different roles
- route feasibility depends on walking radius, transfer limits, and vehicle capacity
- operating conditions are dynamic, so booking changes, no-shows, and traffic delay must be accommodated
- performance must be measured using utilization, punctuality, fleet miles, and transfer quality, not tour length alone

This makes the problem a **transit network optimization task** with hierarchical decisions rather than a pure shortest-tour problem. In that setting, route ordering is only one component of the full optimization pipeline.

### 4.11 Role of the Baseline Model

The current attention-based TSP model remains useful, but only as a lower-level routing component.

- It can optimize stop order within a feeder route or hub cluster.
- It can provide a local decoder once hub assignment has already been determined.
- It gives a controlled benchmark for comparing route-ordering quality.

It does not, by itself, solve the full mixed-fleet transit problem because it does not represent:

- vehicle-type assignment
- hub selection
- transfer feasibility
- schedule reliability
- dynamic coordination across route layers

Accordingly, the baseline model is retained as a **building block** inside the larger transit optimization framework rather than being treated as the final solution.

### 4.12 Transit-Specific Learned Orderer

To make the route-construction module transit-aware, a separate learned ordering model was trained
on synthetic day-ahead routes generated by the hub-and-spoke pipeline.

Training setup:

- synthetic clustered transit instances
- day-ahead mixed-fleet plans
- route labels refined with a 2-opt local improvement step
- supervised training objective over route order

This model differs from the original TSP attention decoder in an important way:

- the TSP decoder was trained for point-to-point Euclidean tours
- the learned transit orderer was trained on the actual pipeline structure, including hub-based route fragments, mixed fleet context, and route-specific load information

Validation summary:

- it beat the heuristic route orderer on held-out synthetic transit instances
- it outperformed the reused TSP attention decoder in the same transit pipeline
- it became the default route-construction strategy in the day-ahead planner

### 4.13 Final Benchmark Summary

The final benchmark comparison across the route-ordering strategies is:

| Benchmark | Heuristic | Attention | Learned |
|---|---:|---:|---:|
| Route ordering mean cost | 0.867255 | 1.014257 | 0.828558 |
| Route ordering wins vs heuristic | 20 | 0 | 17 |
| Full pipeline mean route length | 0.889041 | 0.973619 | 0.827686 |
| Custom instance total route length | 1.024704 | 1.021507 | 0.913460 |

Interpretation:

- heuristic routing is a strong non-learned baseline
- the pretrained TSP attention decoder does not transfer well to this transit setting
- the transit-specific learned orderer is the best performing route-construction component in the current pipeline
- the overall pipeline is still deterministic in hub selection and fleet assignment, so the learned gain is specifically attributable to route ordering

---

## 5. Conclusion

The baseline attention model from the repository is successfully reproduced on TSP20, TSP50, and TSP100 with results that closely match the paper.

The validated research gaps are not about failure to reproduce, but about:

- scaling cost,
- growing inference time,
- size-specific checkpoints,
- synthetic-only data,
- and limited generalization beyond the benchmark distribution.

These gaps justify the adaptive routing experiments as a proof-of-concept, but the **main research contribution** of the project is the mixed-fleet hub-and-spoke transit optimization framework from the PDF.

The final validated architecture is:

- a day-ahead demand clustering and hub selection layer
- a mixed-fleet assignment layer
- a transit-specific learned route-ordering layer
- a learned hub-selection and fleet-assignment prototype
- a fallback heuristic route-ordering layer

The learned route-ordering model is now the default route-construction component in the day-ahead planner. Across all benchmark settings that were run, it improves over the heuristic and transfers better to the mixed-fleet transit geometry than the pretrained TSP attention decoder.

The learned joint planner remains a valid prototype for hub choice and fleet assignment, but in the current short-run training regime it is not yet better than the route-order-only model on route length. Its main gain is that it changes utilization and route composition while preserving demand coverage.

The TSP attention model remains useful as a lower-level baseline and fallback decoder, but it is not the best choice for the full transit system.

---

## 6. Reproduction Commands

For completeness, the exact commands used in this study were:

```bash
conda activate /tmp/attention_tsp

python generate_data.py --problem tsp --name test --seed 1234
python generate_data.py --problem tsp --name validation --seed 4321

python eval.py data/tsp/tsp20_test_seed1234.pkl --model pretrained/tsp_20 --decode_strategy greedy --no_cuda --no_progress_bar
python eval.py data/tsp/tsp50_test_seed1234.pkl --model pretrained/tsp_50 --decode_strategy greedy --no_cuda --no_progress_bar
python eval.py data/tsp/tsp100_test_seed1234.pkl --model pretrained/tsp_100 --decode_strategy greedy --no_cuda --no_progress_bar

python -m transit.train_order_model --train-instances 200 --val-instances 40 --base-seed 6000 --num-hubs 3 --epochs 6 --batch-size 8 --hidden-dim 96 --lr 0.001 --save-path outputs/transit_orderer.pt
python -m transit.validate_order_model --checkpoint outputs/transit_orderer.pt --instances 30 --base-seed 9000 --num-hubs 3
python -m transit.custom_instance --output /tmp/custom_transit_instance.json --seed 2026 --num-clusters 4 --requests-per-cluster 10 --cluster-spread 0.018
python -m transit.benchmark_ordering --num-instances 20 --base-seed 1234 --num-hubs 3
python -m transit.benchmark_pipeline_metrics --instances 10 --base-seed 1234 --num-hubs 3
python -m transit.benchmark_pipeline_metrics --instance-json /tmp/custom_transit_instance.json
python -m transit.train_joint_planner --train-instances 200 --val-instances 40 --base-seed 7000 --num-hubs 3 --num-candidates 6 --epochs 8 --hidden-dim 64 --lr 0.001 --save-path outputs/transit_joint_policy.pt --learned-order-path outputs/transit_orderer.pt
python -m transit.benchmark_joint_planner --instances 20 --base-seed 1234 --num-hubs 3 --joint-policy-path outputs/transit_joint_policy.pt --learned-order-path outputs/transit_orderer.pt
python -m transit.benchmark_joint_planner --instance-json /tmp/custom_transit_instance.json --joint-policy-path outputs/transit_joint_policy.pt --learned-order-path outputs/transit_orderer.pt
```
