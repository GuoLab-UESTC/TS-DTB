# TS-DTB: Two-Stage Multimodal Digital Twin Brain Modeling Framework

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23660150.svg)](https://doi.org/10.5281/zenodo.23660150)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2022b%2B-orange.svg)](https://www.mathworks.com/)

## Overview

**TS-DTB** (Two-Stage Digital Twin Brain) is an individualized, multimodal whole-brain computational modeling framework that integrates structural MRI (dMRI), functional MRI (BOLD fMRI), and high-density electroencephalography (EEG) to infer regional excitation–inhibition (E–I) microcircuit parameters across the human cortex.

The framework operates in two sequential optimization stages:
1. **Stage 1 (Effective Connectivity Iteration):** Starting from empirical structural connectivity (SC), coupled Wilson–Cowan neural mass oscillators and Balloon–Windkessel hemodynamic models are simulated to iteratively refine whole-brain Effective Connectivity (EC) via a time-decayed adaptive learning rate strategy constrained by empirical fMRI functional connectivity (Pearson correlation and Kolmogorov–Smirnov distance).
2. **Stage 2 (Bayesian Parameter Optimization):** Using the individualized EC inferred from Stage 1, source-level neural firing rate signals are projected to the scalp EEG sensor space via individual Finite  Element Method (BEM) leadfields. High-dimensional regional microcircuit parameters, including local excitatory gain ($c_{EE}$), inhibitory gain ($c_{IE}$), background input ($I_b$), and intrinsic noise ($\sigma$) parameterized along functional connectivity gradients, are optimized via Gaussian Process Bayesian Optimization constrained by sensor-level EEG temporal dynamics (**Catch22** feature set) and spectral periodic/aperiodic properties (**FOOOF** / `specparam`).

---

## Repository Structure

```text
TS-DTB/
├── main_TSDTB_demo.m                  # Main entry script (supports TSDTB, EEGOpt, fMRIOpt, rev-TSDTB)
├── run_EC_Iteration.m                 # Stage 1: Iterative Effective Connectivity (EC) optimization
├── run_bayesopt.m                     # Stage 2: Bayesian hyperparameter optimization under EEG constraints
├── TSDTB_Core.m                       # Core static class (Wilson-Cowan, Balloon-Windkessel, FOOOF, Losses)
├── test_data/
│   ├── test_data_HC.mat               # Demo dataset for a representative Healthy Control (test data from BrainLat)
│   └── test_data_AD.mat               # Demo dataset for a representative AD patient (test data from BrainLat)
├── test_results/                      # Output directory for saved optimization results
├── CITATION.cff                       # Citation metadata for GitHub and Zenodo integration
├── LICENSE                            # MIT Open-Source License
└── README.md                          # Documentation and user guide
```

---

## System Requirements & Dependencies

### Software & MATLAB Toolboxes
* **MATLAB** R2022b or later (tested on Windows 10/11 and Linux Ubuntu 20.04/22.04 HPC environments).
* **Statistics and Machine Learning Toolbox** (required for `bayesopt`, `optimizableVariable`, `kstest2`).
* **Signal Processing Toolbox** (required for `pwelch`, `butter`, `filtfilt`, `findpeaks`, `bandpower`).
* **Parallel Computing Toolbox** (optional but recommended for multi-core acceleration via `parpool`).

### External Open-Source Toolboxes
Please ensure the following third-party toolboxes are downloaded and added to your MATLAB path in `main_TSDTB_demo.m`:
1. **EEGLAB** (v2021.0 or later, for FIR filtering `pop_eegfiltnew`): [https://sccn.ucsd.edu/eeglab/](https://sccn.ucsd.edu/eeglab/)
2. **catch22** (MATLAB MEX wrapper for 22 canonical time-series features): [https://github.com/DynamicsAndNeuralSystems/catch22](https://github.com/DynamicsAndNeuralSystems/catch22)
3. **fooof_mat** (MATLAB wrapper for Fitting Oscillations & One-Over-F): [https://github.com/fooof-tools/fooof_mat](https://github.com/fooof-tools/fooof_mat)

---

## Quick Start & Demo Usage

### 1. Configure Paths
Open `main_TSDTB_demo.m` in MATLAB and verify the external toolbox paths at the top of the script:

```matlab
addpath(genpath('./eeglab2025.0'));
addpath(genpath('./catch22'));
addpath(genpath('./fooof_mat-main'));
```

### 2. Select Experimental Mode
Select the target subject group (`demo_type`) and modeling architecture (`Module`) in `main_TSDTB_demo.m`:

```matlab
demo_type        = 'HC';      % Options: 'HC' (sub-CLB00015) or 'AD' (sub-CLB00052)
Module           = 'TSDTB';   % Options: 'TSDTB', 'EEGOpt', 'fMRIOpt', or 'rev-TSDTB'
use_preloaded_EC = false;     % Set to true in 'TSDTB' mode to skip Stage 1 and use pre-computed EC
num_workers      = 8;         % Number of CPU cores for parallel evaluation
```

The `Module` variable directly reproduces the main framework and the ablation comparison models described in our study:
* **`'TSDTB'` (Default Two-Stage Framework):** Runs Stage 1 to infer individualized `EC_opt` from fMRI FC, then executes Stage 2 Bayesian optimization constrained by EEG Catch22 and PSD features. (Setting `use_preloaded_EC = true` loads the pre-converged Stage 1 `EC` directly from `test_data_*.mat` for rapid testing of Stage 2).
* **`'EEGOpt'` (Single-Stage EEG Optimization):** Bypasses Stage 1 and directly uses normalized empirical structural connectivity (`SC_emp_norm`) during Stage 2 Bayesian optimization.
* **`'fMRIOpt'` (Single-Stage fMRI Optimization):** Executes Stage 1 EC iteration using baseline microcircuit parameters without Stage 2 EEG refinement.
* **`'rev-TSDTB'` (Reversed Two-Stage Optimization):** First executes Stage 2 Bayesian optimization using empirical SC, maps the optimal regional parameters to the cortex, and subsequently runs Stage 1 EC iteration.

### 3. Run the Demo
Execute the script in the MATLAB Command Window:

```matlab
>> main_TSDTB_demo
```

### 4. Expected Output
* **Console Output:** Real-time logs displaying Stage 1 iteration progress (mean firing rate, decayed/adaptive learning rate, simulated-empirical FC Pearson correlation, and KS distance) followed by the Stage 2 Bayesian optimization trace and optimal hyperparameters.
* **Saved Results:** Optimization outputs are automatically saved to `./test_results/testRes_<Module>_<demo_type>.mat`, containing:
  * `params_set_opt`: Complete model structure updated with optimal regional parameters ($c_{EE}, c_{IE}, I_b, \sigma, \tau_E, \tau_I, G_e$).
  * `optimal_params`: Best-fitting hyperparameter vector.
  * `EC_opt`: Optimized $200 \times 200$ Effective Connectivity matrix (Schaefer-200 parcellation).
  * `iter_results` & `bayes_results`: Full convergence histories for Stage 1 and Stage 2.

---

## Reproducibility

To guarantee strict numerical reproducibility across runs, subject-specific random seeds (`params_set.rng_seed`) and independent `mrg32k3a` pseudorandom number streams (`RandStream`) are explicitly initialized prior to both Stage 1 and Stage 2 executions.

---

## Citation

If you use this code or the TS-DTB framework in your research, please cite our paper and the archived Zenodo repository:
> Xiong, Y. et al. An EEG-fMRI Jointly Constrained Digital Twin Brain and Its Application in Alzheimer’s Disease. bioRxiv https://doi.org/10.64898/2026.04.10.717365 (2026) doi:10.64898/2026.04.10.717365.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.