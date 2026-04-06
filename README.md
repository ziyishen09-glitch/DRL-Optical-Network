<<<<<<< HEAD
# DRL-based RWTA for Optical Networks

This project is a realization for DRL based Routing, wavelength and timeslot assignment.

## 1. Project Introduction

This project utilizes the PPO model to solve the RWTA problem for optical networks.  
It models dynamic traffic requests under link-failure scenarios, and compares DRL-based decision making with heuristic baselines.

In each decision step, the agent receives a feature representation of the current network and candidate paths, then selects a path-wavelength-timeslot style action (with runtime validity masking). The environment executes allocation, updates traffic occupancy, and returns reward feedback.

Core design ideas:

- State space (observation):
	- Candidate-path related features (path hops, resource occupancy, path availability).
	- Current request information (source, destination, demand/lifetime related context).
	- Network resource tensors (wavelength and timeslot availability snapshots).
	- Failure-aware context in failure-topology simulation.

- Action space:
	- Discrete action over candidate decisions (path and resource selection).
	- Invalid actions are filtered by action masks (Maskable PPO workflow).
	- A safe fallback path is used when all masked logits are invalid.

- Reward design:
	- Positive reward for successful request provisioning.
	- Penalty for blocked requests.
	- Optional shaping terms can favor lower congestion and better resource usage.
	- Long-horizon objective is lower blocking probability and stable performance under failure and load variation.

## 2. Repository Structure and Main Files

### Entry scripts

- PPO_EVAL_STARTER.py
	- Main entry to evaluate trained PPO policy.
	- Supports SB3 backend and ONNX backend.
	- Supports failure lookup table loading/precompute options.

- run_quick_sim.py
	- Lightweight heuristic simulation runner with embedded config.
	- Good for quick baseline checks and failure-link experiments.

- build_failure_lookup_table.py
	- Precomputes impacted source-destination pairs for failed links and K shortest paths.
	- Saves reusable lookup JSON for faster SBP-related checks.

- export_ppo_policy_onnx.py
	- Exports trained PPO policy to ONNX for faster inference in evaluation.
	- Supports Dict observations used by MultiInput policy.

### Core package

- rwa_wdm/BASE_env_online.py
	- Online RWTA environment implementation for step-by-step simulation and feature extraction.

- rwa_wdm/BASE_PPO_EVAL.py
	- PPO evaluation loop, metrics collection, backend switch (SB3/ONNX), and masking logic.

- rwa_wdm/BASE_Heuristic.py
	- Heuristic baseline simulator for comparison.

- rwa_wdm/RWA_functions/traffic_matrix_update.py
	- Traffic/resource matrix update logic when time advances and requests expire.

- rwa_wdm/util.py
	- Common utilities (argument validation, failure-link helpers, lookup table build/load/save helpers).

### Training-related

- PPO/PPO_train_online.py
	- Online PPO training script.

- PPO/PPO_train_v1_offline.py
	- Offline-style PPO training pipeline.

## 3. How to Use

### 3.1 Environment setup

1. Create and activate virtual environment.
2. Install dependencies from dev-requirements.txt.

Example:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r dev-requirements.txt
```

### 3.2 Quick heuristic baseline run

```powershell
python run_quick_sim.py
```

Edit configuration directly in run_quick_sim.py (topology, channels, load range, failure_link, lookup path, etc.).

### 3.3 PPO evaluation run

```powershell
python PPO_EVAL_STARTER.py
```

You can configure:

- inference backend (sb3 or onnx)
- onnx_model_path
- failure_lookup_path / precompute_failure_lookup
- load, calls, num_sim and other simulation parameters

### 3.4 Build failure lookup table

```powershell
python build_failure_lookup_table.py --topology COST239_Failure --channels 8 --failure-link 1 3 --k 3 --output results_ppo/failure_lookup_COST239_1-3_k3.json
```

### 3.5 Export PPO policy to ONNX

```powershell
python export_ppo_policy_onnx.py --model-path <your_model_zip> --output-path <your_output_onnx>
```

## 4. Recent Optimization Focus (Theme of this GitHub Push)

This push focuses on solving slow evaluation runtime in DRL inference and environment stepping.

### 4.1 Problem before optimization

- DRL evaluation was significantly slower than heuristic baseline.
- Main bottlenecks were not only policy forward pass, but also Python-side environment computation:
	- repeated deep-copy style mask-state construction
	- high-overhead traffic matrix update loops
	- repeated SBP impact checks without cached lookup

### 4.2 What was optimized

1. ONNX inference backend integration in evaluation
- Added ONNX Runtime inference path in BASE_PPO_EVAL.py.
- Kept runtime action masking semantics consistent with Maskable PPO.

2. SBP/failure lookup table acceleration
- Added lookup precompute/load/save pipeline.
- Replaced repeated impact checks with cached table queries when available.

3. Environment-side mask generation optimization
- Refactored expensive repeated deep-copy workflow.
- Built per-step slot availability snapshots once and reused for candidate evaluation.

4. Vectorized traffic matrix update
- Introduced edge-index caching and vectorized updates.
- Preserved semantics for occupancy and availability after correctness fix.

### 4.3 Final performance achieved

Under DRL 10 x 10000 evaluation setting:

- DRL average runtime after optimization: about 100 seconds
- Heuristic average runtime: about 30 seconds
- DRL average runtime before optimization: about 478 seconds

This corresponds to a major reduction in DRL evaluation latency while keeping DRL policy behavior and comparison workflow intact.

## 5. Output and Plotting

- Result files are saved under results and results_ppo.
- plotter.py supports metric plotting including SBP-related curves.
- You can use the plotting scripts to compare DRL and heuristic on BP/IT/SBP/utilization trends.

## 6. Notes

- If lookup JSON path points to an existing directory by mistake, generation will fail by design.
- For fair DRL-vs-heuristic comparison, keep topology/failure/load/calls settings aligned.
=======
# DRL-based RWA for QKD-secured optical network 

This repository contains code for DRL-based RWA and link failure recovery in QKD-optical networks.
>>>>>>> 309a74d505d1aa87510b44c04f3ed4692b388b18
