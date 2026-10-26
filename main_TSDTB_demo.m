% MAIN_TSDTB_DEMO Pipeline Demo for TS-DTB Model and Comparison Variants
% Supports four experimental modules described in the manuscript:
%   - 'TSDTB'     : Stage 1 (EC Iteration) -> Stage 2 (Bayesian Opt with EC_opt)
%   - 'EEGOpt'    : Stage 2 only (Bayesian Opt using empirical SC)
%   - 'fMRIOpt'   : Stage 1 only (EC Iteration using baseline parameters)
%   - 'rev-TSDTB' : Stage 2 (using empirical SC) -> Stage 1 (using Stage 2 optimal params)

clear; clc; close all;

%% 0. Add External Toolboxes to Path
addpath(genpath('./eeglab2025.0'));
addpath(genpath('./catch22'));
addpath(genpath('./fooof_mat-main'));

%% 1. Configuration & Data Loading
demo_type        = 'AD';      % Choose 'HC' or 'AD'
Module           = 'TSDTB';   % Choose 'TSDTB', 'EEGOpt', 'fMRIOpt', or 'rev-TSDTB'
use_preloaded_EC = true;     % Set true in 'TSDTB' mode to load pre-computed Stage 1 EC
num_workers      = 32;         % Number of parallel workers (e.g., 8 for local PC, 32 for HPC)

data_file = sprintf('./test_data/test_data_%s.mat', demo_type);
if ~exist(data_file, 'file')
    error('Demo data file not found: %s', data_file);
end
load(data_file, 'params_set');

fprintf('======================================================\n');
fprintf('Loaded Subject : %s (%s)\n', params_set.run_informs.subName, demo_type);
fprintf('Selected Module: %s\n', Module);
fprintf('======================================================\n');

% Initialize output variables
EC_opt         = [];
iter_results   = [];
bayes_results  = [];
optimal_params = struct();
params_set_opt = params_set;

% Start parallel pool if Stage 2 is required
if any(strcmpi(Module, {'TSDTB', 'EEGOpt', 'rev-TSDTB'})) && params_set.optSetting.UseParallel
    current_pool = gcp('nocreate');
    if isempty(current_pool)
        parpool('local', num_workers);
    end
end

%% 2. Execute Selected Modeling Module
switch upper(Module)
    
    case 'TSDTB'
        % --- Step 1: Obtain Effective Connectivity (EC) ---
        if ~use_preloaded_EC
            fprintf('\n>>> [TSDTB] Running Stage 1: EC Iteration...\n');
            rng(params_set.rng_seed, 'twister');
            params_set.r_noise = {RandStream.create('mrg32k3a', 'Seed', params_set.rng_seed)};
            [EC_opt, iter_results] = run_EC_Iteration(params_set);
        else
            fprintf('\n>>> [TSDTB] Skipping Stage 1. Loading pre-computed EC from %s...\n', data_file);
            tmp = load(data_file, 'EC');
            EC_opt = tmp.EC;
        end
        params_set.EC = EC_opt;
        
        % --- Step 2: Bayesian Optimization under EEG Constraints ---
        fprintf('\n>>> [TSDTB] Running Stage 2: Bayesian Optimization...\n');
        rng(params_set.rng_seed, 'twister');
        params_set.r_noise = {RandStream.create('mrg32k3a', 'Seed', params_set.rng_seed)};
        [bayes_results, optimal_params] = run_bayesopt(params_set);
        params_set_opt = TSDTB_Core.mapBayesianParams(optimal_params, params_set);

    case 'EEGOPT'
        % --- Stage 2 Only: Use empirical SC directly ---
        fprintf('\n>>> [EEGOpt] Using Empirical SC for Stage 2 Bayesian Optimization...\n');
        SC_emp = params_set.SC_emp_norm;
        params_set.EC = SC_emp - diag(diag(SC_emp));
        
        rng(params_set.rng_seed, 'twister');
        params_set.r_noise = {RandStream.create('mrg32k3a', 'Seed', params_set.rng_seed)};
        [bayes_results, optimal_params] = run_bayesopt(params_set);
        params_set_opt = TSDTB_Core.mapBayesianParams(optimal_params, params_set);

    case 'FMRIOPT'
        % --- Stage 1 Only: Iterative EC optimization ---
        fprintf('\n>>> [fMRIOpt] Running Stage 1: EC Iteration Only...\n');
        rng(params_set.rng_seed, 'twister');
        params_set.r_noise = {RandStream.create('mrg32k3a', 'Seed', params_set.rng_seed)};
        [EC_opt, iter_results] = run_EC_Iteration(params_set);
        params_set_opt.EC = EC_opt;

    case 'REV-TSDTB'
        % --- Step 1: Run Stage 2 first using empirical SC ---
        fprintf('\n>>> [rev-TSDTB] Step 1: Running Stage 2 with Empirical SC...\n');
        SC_emp = params_set.SC_emp_norm;
        params_set.EC = SC_emp - diag(diag(SC_emp));
        
        rng(params_set.rng_seed, 'twister');
        params_set.r_noise = {RandStream.create('mrg32k3a', 'Seed', params_set.rng_seed)};
        [bayes_results, optimal_params] = run_bayesopt(params_set);
        params_set_opt = TSDTB_Core.mapBayesianParams(optimal_params, params_set);
        
        % --- Step 2: Run Stage 1 EC Iteration with updated regional parameters ---
        fprintf('\n>>> [rev-TSDTB] Step 2: Running Stage 1 EC Iteration with optimized parameters...\n');
        rng(params_set_opt.rng_seed, 'twister');
        params_set_opt.r_noise = {RandStream.create('mrg32k3a', 'Seed', params_set_opt.rng_seed)};
        [EC_opt, iter_results] = run_EC_Iteration(params_set_opt);
        params_set_opt.EC = EC_opt;

    otherwise
        error('Unknown Module: %s. Choose TSDTB, EEGOpt, fMRIOpt, or rev-TSDTB.', Module);
end

%% 3. Save Results
out_dir = './test_results';
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end
out_file = fullfile(out_dir, sprintf('testRes_%s_%s.mat', Module, demo_type));
save(out_file, 'Module', 'demo_type', 'params_set_opt', 'optimal_params', ...
               'EC_opt', 'iter_results', 'bayes_results', '-v7.3');

%% 4. Final Summary Output
fprintf('\n================ OPTIMIZATION COMPLETE ================\n');
fprintf('Results saved to: %s\n', out_file);
if ~isempty(fieldnames(optimal_params))
    disp('Optimal Hyperparameters:');
    disp(optimal_params);
end