function [bayes_results, optimalParams] = run_bayesopt(params_set)
% run_bayesopt Stage 2 of the TS-DTB framework.
%   Optimizes global coupling, time constants, and gradient-weighted regional 
%   heterogeneity parameters under sensor-level EEG temporal (Catch22) and 
%   spectral (PSD/FOOOF) constraints using Bayesian optimization.
%
% Inputs:
%   params_set    - Structure containing model priors, empirical features, 
%                   effective connectivity (params_set.EC), and optimizer settings.
%
% Outputs:
%   bayes_results - BayesianOptimization object containing optimization trace.
%   optimalParams - Structure of best-fitting hyperparameters at minimum objective.

    fprintf('\n--- [Stage 2] Executing Bayesian Optimization ---\n');
    
    % 1. Dynamically define optimizable variables from configuration
    opt_params = [];
    bayes_params = fieldnames(params_set.opt_variable_set);
    for i = 1:numel(bayes_params)
        params_name = bayes_params{i};
        params_range = params_set.opt_variable_set.(params_name);
        opt_params = [opt_params, optimizableVariable(params_name, params_range, 'Type', 'real')];
    end
    
    opts = params_set.optSetting;
    fprintf('Starting Bayesian Optimization (%d variables, MaxEvaluations = %d)...\n', ...
        length(bayes_params), opts.MaxObjectiveEvaluations);
    
    % 2. Define objective function handle
    objectiveFcn = @(opt_vars) evaluate_EEG_Loss(opt_vars, params_set);
    
    % 3. Execute Bayesian Optimization
    bayes_results = bayesopt(objectiveFcn, opt_params, ...
        'NumSeedPoints', opts.NumSeedPoints, ...
        'MaxObjectiveEvaluations', opts.MaxObjectiveEvaluations, ...
        'AcquisitionFunctionName', opts.AcquisitionFunctionName, ...
        'ExplorationRatio', opts.ExplorationRatio, ...
        'GPActiveSetSize', opts.GPActiveSetSize, ...
        'UseParallel', opts.UseParallel, ...
        'Verbose', 1);
    
    % 4. Extract optimal hyperparameters
    optimalParams = struct();
    for i = 1:numel(bayes_params)
        params_name = bayes_params{i};
        optimalParams.(params_name) = bayes_results.XAtMinObjective.(params_name);
    end
    
    fprintf('--- Stage 2 Complete (Min Objective Loss: %.4f) ---\n', bayes_results.MinObjective);
end

function obj_val = evaluate_EEG_Loss(opt_vars, params_set)
% EVALUATE_EEG_LOSS Objective function evaluating simulated EEG Catch22 and PSD features.

    % 1. Map Bayesian hyperparameters to regional brain parameters
    params_set = TSDTB_Core.mapBayesianParams(opt_vars, params_set);
    
    % 2. Simulate neural mass dynamics (BOLD simulation disabled in Stage 2)
    [rE_signals, ~, ~, ~] = TSDTB_Core.simulateDynamics(params_set, params_set.EC, params_set.G_e, false);
    
    % 3. Project source activity to EEG sensors via Leadfield, bandpass filter, and Z-score
    eeg_signals = params_set.leadfield * rE_signals;
    eeg_filt    = TSDTB_Core.filterEEG(eeg_signals, params_set);
    eeg_z       = (eeg_filt - mean(eeg_filt, 2)) ./ std(eeg_filt, 0, 2);
    
    % 4. Compute Sensor-Level EEG Catch22 Loss
    loss_catch22 = 0;
    if params_set.loss_list.eegcatch22 > 0
        nCh = size(eeg_z, 1);
        sim_catch22 = zeros(nCh, 24);
        parfor ch = 1:nCh
            [sim_catch22(ch, :), ~, ~] = catch22_all(double(eeg_z(ch, :)'));
        end
        loss_catch22 = TSDTB_Core.computeCatch22Loss(params_set.EEGcatch22_emp(:, 1:22), sim_catch22(:, 1:22));
    end
    
    % 5. Compute Sensor-Level EEG Spectral (PSD & FOOOF) Loss
    loss_psd = 0;
    if params_set.loss_list.eegSpectral > 0
        psd_emp = params_set.psd_emp;
        if isfield(psd_emp, 'psd_freqs_emp')
            freqs_target = psd_emp.psd_freqs_emp;
        else
            freqs_target = psd_emp.freqs;
        end
        
        % Compute simulated PSD and FOOOF decomposition
        sim_psd = TSDTB_Core.computePSD(eeg_z, params_set.resampled, params_set.filterBand, freqs_target);
        
        % Evaluate comprehensive PSD feature losses (aligned with psd_features_fitlosses.m)
        psd_fitting_results = TSDTB_Core.computePSDLoss(psd_emp, sim_psd, [], [], 'mean', true);
        
        loss_freqsfields = { ...
            'lossPearson', 'MeanAbs', ...
            'exponent', 'offset', ...
            'peakFreq', 'peakPower', 'peakRatio', 'peakWidth', 'peakHeight', 'peakQ', 'peakEnergy1', ...
            'deltaEnergy1', 'thetaEnergy1', 'alphaEnergy1', 'betaEnergy1', 'gammaEnergy1' ...
        };
        
        loss_freqs = cellfun(@(f) psd_fitting_results.(f), loss_freqsfields);
        loss_psd   = median(loss_freqs, 'omitnan');
    end
    
    % 6. Aggregate weighted objective value
    obj_val = (params_set.loss_list.eegcatch22 * loss_catch22) + ...
              (params_set.loss_list.eegSpectral * loss_psd);
end