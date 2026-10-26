classdef TSDTB_Core
    % TSDTB_CORE Core static methods for the TS-DTB computational modeling framework.
    %   Encapsulates parameter mapping, Wilson-Cowan neural mass simulation, 
    %   Balloon-Windkessel BOLD hemodynamics, EEGLAB FIR filtering, FOOOF spectral 
    %   decomposition, and multi-modal objective loss evaluation.
    
    methods (Static)
        
        %% ================= 1. Parameter Mapping & Updating =================
        function params_set = mapBayesianParams(opt_params, params_set)
            % Maps Bayesian optimization hyperparameters to regional brain parameters
            if istable(opt_params)
                opt_struct = table2struct(opt_params);
            elseif isstruct(opt_params)
                opt_struct = opt_params;
            else
                pNames = fieldnames(params_set.opt_variable_set);
                opt_struct = cell2struct(num2cell(opt_params(:)), pNames, 1);
            end
            
            % Copy optimized fields into params_set
            fnames = fieldnames(opt_struct);
            for i = 1:length(fnames)
                params_set.(fnames{i}) = opt_struct.(fnames{i});
            end
            
            % Update global coupling
            if isfield(opt_struct, 'G_e'), params_set.G_e = opt_struct.G_e; end
            if isfield(opt_struct, 'G_i'), params_set.G_i = opt_struct.G_i; end
            
            % Cortical FC Gradients
            G1 = params_set.fcGradients1;
            G2 = params_set.fcGradients2;
            
            % A. Synaptic Weights (cEE, cIE, cEI, cII)
            thres_c = [1, 50];
            synaptic_targets = {'cIE', 'cEE', 'cEI', 'cII'};
            for i = 1:length(synaptic_targets)
                target = synaptic_targets{i};
                if isfield(opt_struct, ['c_', target])
                    a = TSDTB_Core.getSafeField(opt_struct, ['a_', target], 0);
                    b = TSDTB_Core.getSafeField(opt_struct, ['b_', target], 0);
                    c = TSDTB_Core.getSafeField(opt_struct, ['c_', target], 0);
                    raw_val = a * G1 + b * G2 + c;
                    params_set.(target) = TSDTB_Core.mapParameter(raw_val, thres_c, 'linear');
                end
            end
            
            % B. Background Input (Ib)
            thres_Ib = [0, 3];
            if isfield(opt_struct, 'c_Ib')
                a = TSDTB_Core.getSafeField(opt_struct, 'a_Ib', 0);
                b = TSDTB_Core.getSafeField(opt_struct, 'b_Ib', 0);
                c = TSDTB_Core.getSafeField(opt_struct, 'c_Ib', 0);
                val = TSDTB_Core.mapParameter(a * G1 + b * G2 + c, thres_Ib, 'linear');
                params_set.Ib_e = val;
                params_set.Ib_i = val;
            end
            
            % C. Noise Level (Sigma)
            thres_Sigma = [1e-4, 1e-1];
            if isfield(opt_struct, 'c_sigma')
                a = TSDTB_Core.getSafeField(opt_struct, 'a_sigma', 0);
                b = TSDTB_Core.getSafeField(opt_struct, 'b_sigma', 0);
                c = TSDTB_Core.getSafeField(opt_struct, 'c_sigma', -3);
                val_sig = TSDTB_Core.mapParameter(a * G1 + b * G2 + c, thres_Sigma, 'log');
                params_set.sigmaEdeep = val_sig;
                params_set.sigmaIdeep = val_sig;
            end
            
            % D. Time Constants (tauE and tauI)
            if isfield(opt_struct, 'tauE')
                if isscalar(opt_struct.tauE)
                    params_set.tauE = opt_struct.tauE * ones(params_set.nROI, 1);
                end
            end
            if isfield(opt_struct, 'r_tauI')
                params_set.tauI = opt_struct.r_tauI .* params_set.tauE;
            end
        end
        
        function val = mapParameter(raw_input, target_range, scale_mode)
            % Smoothly restricts values to physiological ranges using hyperbolic tangent
            lower = target_range(1);
            upper = target_range(2);
            if strcmpi(scale_mode, 'linear')
                center = (upper + lower) / 2;
                half_width = (upper - lower) / 2;
                val = center + half_width * tanh((raw_input - center) / half_width);
            elseif strcmpi(scale_mode, 'log')
                log_lower = log10(lower);
                log_upper = log10(upper);
                center = (log_upper + log_lower) / 2;
                half_width = (log_upper - log_lower) / 2;
                log_val = center + half_width * tanh((raw_input - center) / half_width);
                val = 10 .^ log_val;
            else
                val = min(max(raw_input, lower), upper);
            end
        end
        
        %% ================= 2. Neural Dynamics & BOLD Simulation =================
        function [rE_resampled, EI_ratio, simFC, bold_metrics] = simulateDynamics(params_set, W1, G_e, simulate_BOLD)
            % Simulates coupled Wilson-Cowan oscillators and optionally BOLD hemodynamics
            % Remove self-connections on the diagonal
            W1 = W1 - diag(diag(W1));
            
            % Time settings
            t_times    = params_set.t_times;
            tcut       = params_set.tcut;
            dt         = params_set.dt;
            time_total = (t_times + tcut) * 1000; % Total ms (recorded at 1000 Hz)
            resampled  = TSDTB_Core.getSafeField(params_set, 'resampled', 250);
            fisherType = TSDTB_Core.getSafeField(params_set, 'fisherType', true);
            normFCType = TSDTB_Core.getSafeField(params_set, 'normFCType', true);

            % Regional Parameters
            nROI       = params_set.nROI;
            tauE       = params_set.tauE;       tauI       = params_set.tauI;
            cEE        = params_set.cEE;        cEI        = params_set.cEI;
            cIE        = params_set.cIE;        cII        = params_set.cII;
            a_e        = params_set.a_e;        theta_e    = params_set.theta_e;
            a_i        = params_set.a_i;        theta_i    = params_set.theta_i;
            S0e        = 1 / (1 + exp(a_e * theta_e));
            S0i        = 1 / (1 + exp(a_i * theta_i));
            Smax_e     = 1 - S0e;               Smax_i     = 1 - S0i;
            Ib_e       = params_set.Ib_e;       Ib_i       = params_set.Ib_i;
            sigmaEdeep = params_set.sigmaEdeep; sigmaIdeep = params_set.sigmaIdeep;
            G_i        = TSDTB_Core.getSafeField(params_set, 'G_i', 0);
            
            % Initialize state variables
            rE = 0.1 * ones(nROI, 1);
            rI = 0.1 * ones(nROI, 1);
            rE_downsampled = 0.1 * ones(time_total, nROI);
            
            j = 0; nn = 1;
            E_sum = zeros(1, nROI);
            I_sum = zeros(1, nROI);
            
            % Obtain random noise stream
            if isfield(params_set, 'r_noise') && ~isempty(params_set.r_noise)
                noise_stream = params_set.r_noise{1};
            else
                noise_stream = RandStream.getGlobalStream();
            end

            % Main Euler-Maruyama Integration Loop (step = dt ms)
            steps_per_ms = round(1 / dt);
            for t = 2 : (time_total / dt)
                InetE = cEE .* rE - cIE .* rI + G_e * (W1 * rE) + Ib_e;
                InetI = cEI .* rE - cII .* rI + G_i * (W1 * rI) + Ib_i;
                
                sigmodialE = 1 ./ (1 + exp(-a_e .* (InetE - theta_e))) - S0e;
                sigmodialI = 1 ./ (1 + exp(-a_i .* (InetI - theta_i))) - S0i;
                
                rE = rE + (dt ./ tauE) .* (-rE + (Smax_e - rE) .* sigmodialE) + ...
                    ((sigmaEdeep ./ sqrt(tauE)) .* sqrt(dt) .* randn(noise_stream, nROI, 1));
                
                rI = rI + (dt ./ tauI) .* (-rI + (Smax_i - rI) .* sigmodialI) + ...
                    ((sigmaIdeep ./ sqrt(tauI)) .* sqrt(dt) .* randn(noise_stream, nROI, 1));
                
                % Record at 1 ms intervals (1000 Hz)
                j = j + 1;
                if j == steps_per_ms
                    rE_downsampled(nn, :) = rE';
                    E_sum = E_sum + max(rE', 0);
                    I_sum = I_sum + max(rI', 0);
                    nn = nn + 1;
                    j = 0;
                end
            end
            
            % Compute E-I ratio (ratio_of_means)
            EI_ratio = (E_sum ./ nn) ./ (I_sum ./ nn);
            
            % Downsample rE from 1000 Hz (post tcut)
            ds_step = floor(1000 / resampled);
            rE_resampled = single(rE_downsampled(tcut*1000 : ds_step : nn-1, :)');
            
            % Simulate BOLD via Balloon-Windkessel model if requested
            simFC = []; bold_metrics = struct();
            if simulate_BOLD
                n_valid = nn - 1;
                Tt = n_valid / 1000;
                simTR = 0.001;
                BOLD_Signal = zeros(n_valid, nROI);
                for i = 1:nROI
                    BOLD_Signal(:, i) = TSDTB_Core.balloonWindkessel(Tt, rE_downsampled(1:n_valid, i)');
                end
                
                % Step 1: Anti-aliasing low-pass filter (< 0.8 * Nyquist of TR)
                target_nyquist = 1 / (2 * params_set.TR);
                low_cut = 0.8 * target_nyquist;
                signal_lp = TSDTB_Core.filterButterworth(BOLD_Signal, simTR, [0, low_cut], 2);
                
                % Step 2: Truncate initial transient (tcut) and downsample to fMRI TR
                signal_ds = signal_lp(tcut*1000 : round(params_set.TR / simTR) : end, :);
                
                % Step 3: Bandpass filter [0.01, 0.10] Hz and Z-score normalize
                simBOLD = zscore(TSDTB_Core.filterButterworth(signal_ds, params_set.TR, params_set.ButterworthBand, 2), 0, 1);
                
                % Compute simulated Functional Connectivity
                simFC = corr(simBOLD);
                simFC = simFC - diag(diag(simFC));
                
                if (islogical(fisherType) && fisherType) || (ischar(fisherType) && strcmpi(fisherType, 'FisherZ'))
                    simFC = atanh(simFC);
                    simFC(1:size(simFC,1)+1:end) = 0;
                end
                if normFCType
                    simFC = simFC ./ max(abs(simFC(:)));
                end
                
                empFC = TSDTB_Core.getSafeField(params_set, 'FC_emp_norm', params_set.FC_emp);
                bold_metrics = TSDTB_Core.computeMatrixFitting(empFC, simFC);
                bold_metrics.RE_mean_ROI = mean(rE_downsampled(1:n_valid, :), 1)';
                bold_metrics.RE_mean     = mean(bold_metrics.RE_mean_ROI);
            end
        end
        
        function b = balloonWindkessel(T, r)
            % Balloon-Windkessel hemodynamic model (Friston et al., 2003)
            dt = 0.001;
            n_t = round(T / dt);
            
            taus = 0.65; tauf = 0.41; tauo = 0.98; alpha = 0.32;
            itaus = 1/taus; itauf = 1/tauf; itauo = 1/tauo; ialpha = 1/alpha;
            Eo = 0.34; vo = 0.02; k1 = 7*Eo; k2 = 2; k3 = 2*Eo - 0.2;
            
            x = zeros(n_t, 4);
            x(1, :) = [0 1 1 1];
            for n = 1 : n_t - 1
                x(n+1,1) = x(n,1) + dt * (r(n) - itaus*x(n,1) - itauf*(x(n,2) - 1));
                x(n+1,2) = x(n,2) + dt * x(n,1);
                x(n+1,3) = x(n,3) + dt * itauo * (x(n,2) - x(n,3)^ialpha);
                x(n+1,4) = x(n,4) + dt * itauo * (x(n,2)*(1 - (1 - Eo)^(1/x(n,2)))/Eo - (x(n,3)^ialpha)*x(n,4)/x(n,3));
            end
            v = x(:, 3);
            q = x(:, 4);
            b = (100/Eo) * vo * (k1.*(1 - q) + k2.*(1 - q./v) + k3.*(1 - v));
        end
        
        %% ================= 3. Signal Filtering & Processing =================
        function sig_filt = filterButterworth(Signals, TR, ButterworthBand, k)
            % Zero-phase Butterworth filter supporting lowpass, highpass, and bandpass
            if nargin < 4, k = 2; end
            fnq = 1 / (2 * TR);
            flp = ButterworthBand(1);
            fhi = ButterworthBand(2);
            
            if flp <= 0 && isfinite(fhi)
                Wn = fhi / fnq;
                [bfilt, afilt] = butter(k, Wn, 'low');
            elseif isfinite(flp) && (~isfinite(fhi) || fhi >= fnq)
                Wn = flp / fnq;
                [bfilt, afilt] = butter(k, Wn, 'high');
            elseif flp > 0 && isfinite(fhi) && flp < fhi
                Wn = [flp/fnq, fhi/fnq];
                [bfilt, afilt] = butter(k, Wn, 'bandpass');
            else
                error('Invalid ButterworthBand: [%g %g] with fnq=%g', flp, fhi, fnq);
            end
            
            sig_filt = filtfilt(bfilt, afilt, double(Signals));
        end
        
        function sig_filt = filterEEG(signals, params_set)
            % Bandpass filters sensor EEG signals using EEGLAB FIR filter (pop_eegfiltnew)
            if size(signals, 1) > size(signals, 2)
                signals = signals';
            end
            
            EEG = struct();
            EEG.data     = signals;
            EEG.srate    = params_set.resampled;
            EEG.nbchan   = min(size(signals));
            EEG.pnts     = max(size(signals));
            EEG.trials   = 1;
            EEG.times    = linspace(0, (EEG.pnts-1) / EEG.srate, EEG.pnts);
            EEG.xmin     = 0;
            EEG.xmax     = (EEG.pnts-1) / EEG.srate;
            EEG.chanlocs = params_set.chanlocs;
            EEG.setname  = 'EEG';
            EEG.comments = ''; EEG.ref = [];
            EEG.icaact = []; EEG.icawinv = []; EEG.icasphere = []; EEG.icaweights = []; EEG.icachansind = [];
            
            evalc('EEG = eeg_checkset(EEG);');
            f_band = params_set.filterBand;
            evalc('EEG = pop_eegfiltnew(EEG, f_band(1), f_band(2));');
            
            sig_filt = single(EEG.data);
        end
        
        %% ================= 4. PSD & FOOOF Computation =================
        function psd = computePSD(signals, fs, f_range, target_freqs)
            % Computes Welch's PSD and decomposes periodic/aperiodic components via FOOOF
            if nargin < 2 || isempty(fs), fs = 250; end
            if nargin < 3 || isempty(f_range), f_range = [0.5, 45]; end
            
            signals = double(signals);
            [nCh, nTime] = size(signals);
            
            winLen  = 2 * floor(fs);
            window  = hanning(winLen);
            nfft    = 2^nextpow2(winLen);
            overlap = winLen / 2;
            
            [~, f_full] = pwelch(randn(nTime, 1), window, overlap, nfft, fs);
            if nargin >= 4 && ~isempty(target_freqs)
                freq_mask = f_full >= min(target_freqs) & f_full <= max(target_freqs);
            else
                freq_mask = f_full >= f_range(1) & f_full <= f_range(2);
            end
            freqs = f_full(freq_mask)';
            
            [pxx, ~] = pwelch(signals', window, overlap, nfft, fs);
            psd_linear = pxx(freq_mask, :)'; % [nCh x nFreqs]
            
            psd = TSDTB_Core.fitMultiPSDFOOOF(freqs, psd_linear, f_range);
            psd.psd_linear       = psd.linear;
            psd.psd_sim_spectopo = psd.log;
            psd.psd_flattened    = psd.flattened;
            psd.psd_aperiodic    = psd.aperiodic;
            psd.aperiodic_params = psd.offset_exp;
            psd.freqs_signals    = freqs;
        end
        
        function psd = fitMultiPSDFOOOF(freqs, psd_linear, f_range)
            % Aligned with get_multiPSD_fooof.m
            if nargin < 3 || isempty(f_range)
                f_range = [min(freqs), max(freqs)];
            else
                f_range = [max(f_range(1), min(freqs)), min(f_range(2), max(freqs))];
            end
            if min(psd_linear(:)) < 0
                psd_linear = 10.^(psd_linear / 10);
            end
            
            [nCh, nFreqs] = size(psd_linear);
            settings = struct('peak_width_limits', [0.5, 12], 'max_n_peaks', 0, ...
                              'peak_threshold', 2, 'min_peak_height', 0.0, ...
                              'aperiodic_mode', 'fixed', 'verbose', false);
            
            fooof_results = fooof_group(freqs, psd_linear', f_range, settings);
            
            psd.linear     = psd_linear;
            psd.log        = nan(nCh, nFreqs);
            psd.flattened  = nan(nCh, nFreqs);
            psd.aperiodic  = nan(nCh, nFreqs);
            psd.offset_exp = nan(nCh, 3);
            psd.freqs      = freqs;
            
            psd_log10 = log10(psd_linear);
            for ch = 1:nCh
                res = fooof_results(ch);
                off = res.aperiodic_params(1);
                exp_val = res.aperiodic_params(2);
                r2 = res.r_squared;
                psd.offset_exp(ch, :) = [off, exp_val, r2];
                
                ap_fit_log = off - exp_val * log10(freqs);
                psd.aperiodic(ch, :) = 10 * ap_fit_log;
                psd.flattened(ch, :) = 10 * (psd_log10(ch, :) - ap_fit_log);
                psd.log(ch, :)       = 10 * psd_log10(ch, :);
            end
        end
        
        function res_ap = fitSingleFOOOF(freqs, psd_vec, f_range)
            % Fits FOOOF on a single 1 x nFreqs PSD vector (aligned with psd_fooof)
            if nargin < 3 || isempty(f_range)
                f_range = [min(freqs), max(freqs)];
            end
            settings = struct('peak_width_limits', [0.5, 12], 'max_n_peaks', 0, ...
                              'peak_threshold', 2, 'min_peak_height', 0.0, ...
                              'aperiodic_mode', 'fixed', 'verbose', false);
            res_ap = fooof(freqs(:), psd_vec(:), f_range, settings, false);
        end
        
        function [bandE_raw, bandE_norm] = calculatePSDEnergy(PSD, freqs, bands)
            % Aligned with calculate_eegPSDEnergy.m
            if min(PSD(:)) < 0
                PSD = 10.^(PSD / 10);
            end
            if nargin < 3 || isempty(bands)
                bands = struct('delta',[0.5 4], 'theta',[4 8], 'alpha',[8 13], ...
                               'beta',[13 30], 'gama',[30 45], 'total',[0.5 45]);
            end
            bandNames = fieldnames(bands);
            nBands = numel(bandNames);
            freqs = freqs(:)';
            nCh = size(PSD, 1);
            
            bandE_raw  = zeros(nCh, nBands);
            bandE_norm = zeros(nCh, nBands);
            for ch = 1:nCh
                psd_raw  = PSD(ch, :);
                psd_norm = (psd_raw - min(psd_raw)) / (max(psd_raw) - min(psd_raw) + eps);
                for i = 1:nBands
                    br = bands.(bandNames{i});
                    br(1) = max(br(1), min(freqs));
                    br(2) = min(br(2), max(freqs));
                    bandE_raw(ch, i)  = bandpower(psd_raw, freqs, br, 'psd');
                    bandE_norm(ch, i) = bandpower(psd_norm, freqs, br, 'psd');
                end
            end
        end
        
        %% ================= 5. Loss & Metrics Evaluation =================
        function loss = computeCatch22Loss(emp, sim)
            % Computes Catch22 median relative error across channels and features
            relErr = TSDTB_Core.relErr(sim, emp);
            relErrFeat = median(relErr, 1, 'omitnan');
            loss = TSDTB_Core.clip01(median(relErrFeat, 'omitnan'));
        end
        
        function results = computePSDLoss(psd_emp, psd_sim, bands, peak_thresh, calc_mode, use_relative)
            % Strictly aligned with psd_features_fitlosses.m
            if nargin < 6 || isempty(use_relative), use_relative = true; end
            if nargin < 5 || isempty(calc_mode),    calc_mode    = 'channel'; end
            if nargin < 4 || isempty(peak_thresh),  peak_thresh  = 0.1; end
            if nargin < 3 || isempty(bands)
                bands = struct('delta',[0.5 4], 'theta',[4 8], 'alpha',[8 13], ...
                               'beta',[13 30], 'gamma',[30 45], 'total',[0.5 45]);
            end
            if isfield(bands, 'gama') && ~isfield(bands, 'gamma')
                bands.gamma = bands.gama;
                bands = rmfield(bands, 'gama');
            end
            
            if isfield(psd_emp, 'psd_freqs_emp')
                freqs = psd_emp.psd_freqs_emp;
            elseif isfield(psd_emp, 'freqs')
                freqs = psd_emp.freqs;
            end
            if isfield(psd_emp, 'log')
                psd_emp.psd_spectopo_emp = psd_emp.log;
            end
            freqs_alpha = bands.alpha;
            
            % Extract linear PSD
            if isfield(psd_emp, 'psd_linear') && ~isempty(psd_emp.psd_linear)
                P_emp_lin_all = psd_emp.psd_linear;
            elseif isfield(psd_emp, 'linear') && ~isempty(psd_emp.linear)
                P_emp_lin_all = psd_emp.linear;
            else
                P_emp_lin_all = TSDTB_Core.fitMultiPSDFOOOF(freqs, 10.^(psd_emp.psd_spectopo_emp / 10)).linear;
            end
            if isfield(psd_sim, 'psd_linear') && ~isempty(psd_sim.psd_linear)
                P_sim_lin_all = psd_sim.psd_linear;
            else
                P_sim_lin_all = psd_sim.linear;
            end
            
            % Extract log PSD (dB)
            if isfield(psd_emp, 'psd_spectopo_emp') && ~isempty(psd_emp.psd_spectopo_emp)
                P_emp_log_all = psd_emp.psd_spectopo_emp;
            else
                P_emp_log_all = 10 * log10(P_emp_lin_all);
            end
            if isfield(psd_sim, 'psd_sim_spectopo') && ~isempty(psd_sim.psd_sim_spectopo)
                P_sim_log_all = psd_sim.psd_sim_spectopo;
            elseif isfield(psd_sim, 'log') && ~isempty(psd_sim.log)
                P_sim_log_all = psd_sim.log;
            else
                P_sim_log_all = 10 * log10(P_sim_lin_all);
            end
            
            if strcmpi(calc_mode, 'mean')
                P_emp_lin = mean(P_emp_lin_all, 1, 'omitnan');
                P_sim_lin = mean(P_sim_lin_all, 1, 'omitnan');
                P_emp_log = mean(P_emp_log_all, 1, 'omitnan');
                P_sim_log = mean(P_sim_log_all, 1, 'omitnan');
                nLoop = 1;
            else
                P_emp_lin = P_emp_lin_all;
                P_sim_lin = P_sim_lin_all;
                P_emp_log = P_emp_log_all;
                P_sim_log = P_sim_log_all;
                nLoop = size(P_emp_lin_all, 1);
            end
            
            peak_fields = {'peakFreq', 'peakPower', 'peakWidth', 'peakHeight', 'peakQ', ...
                           'peakRatio', 'peakEnergy', 'peakEnergy1'};
            bandNames = fieldnames(bands);
            bandNames(strcmpi(bandNames, 'total')) = [];
            band_fields = {};
            for ii = 1:numel(bandNames)
                band_fields{end+1} = [bandNames{ii}, 'Energy'];
                band_fields{end+1} = [bandNames{ii}, 'Energy1'];
            end
            all_loss_fields = [peak_fields, band_fields];
            
            dummy_loss = cell2struct(num2cell(nan(1, numel(all_loss_fields))), all_loss_fields, 2);
            loop_res_arr = repmat(dummy_loss, nLoop, 1);
            
            for i = 1:nLoop
                if nLoop == 1
                    P_e_log_curr = P_emp_log; P_s_log_curr = P_sim_log;
                    P_e_lin_curr = P_emp_lin; P_s_lin_curr = P_sim_lin;
                else
                    P_e_log_curr = P_emp_log(i, :); P_s_log_curr = P_sim_log(i, :);
                    P_e_lin_curr = P_emp_lin(i, :); P_s_lin_curr = P_sim_lin(i, :);
                end
                
                if use_relative
                    P_e_lin_curr = (P_e_lin_curr - min(P_e_lin_curr)) / (max(P_e_lin_curr) - min(P_e_lin_curr));
                    P_s_lin_curr = (P_s_lin_curr - min(P_s_lin_curr)) / (max(P_s_lin_curr) - min(P_s_lin_curr));
                    P_e_log_curr = (P_e_log_curr - min(P_e_log_curr)) / (max(P_e_log_curr) - min(P_e_log_curr));
                    P_s_log_curr = (P_s_log_curr - min(P_s_log_curr)) / (max(P_s_log_curr) - min(P_s_log_curr));
                end
                
                [pks_e, locs_e, widths_e, proms_e] = findpeaks(P_e_log_curr, freqs, ...
                    'MinPeakProminence', peak_thresh, 'MinPeakDistance', 6, ...
                    'MinPeakWidth', 1, 'MaxPeakWidth', 12, 'NPeaks', 3);
                [pks_s, locs_s, widths_s, proms_s] = findpeaks(P_s_log_curr, freqs, ...
                    'MinPeakProminence', peak_thresh, 'MinPeakDistance', 6, ...
                    'MinPeakWidth', 1, 'MaxPeakWidth', 12, 'NPeaks', 3);
                
                has_e = ~isempty(pks_e);
                has_s = ~isempty(pks_s);
                res = dummy_loss;
                
                if has_e
                    alpha_mask = locs_e >= freqs_alpha(1) & locs_e <= freqs_alpha(2);
                    if any(alpha_mask)
                        idx_list = find(alpha_mask);
                        [~, sub_idx] = max(pks_e(alpha_mask));
                        idx_e = idx_list(sub_idx);
                    else
                        [~, idx_e] = max(pks_e);
                    end
                    emp_freq  = locs_e(idx_e);
                    emp_val   = pks_e(idx_e);
                    emp_width = widths_e(idx_e);
                    emp_prom  = proms_e(idx_e);
                    emp_Q     = emp_freq / (emp_width + eps);
                    tb = [max(emp_freq - emp_width/2 - 1, freqs(1)), ...
                          min(emp_freq + emp_width/2 + 1, freqs(end))];
                    if use_relative
                        [~, idx_closest] = min(abs(freqs - emp_freq));
                        peak_lin = P_e_lin_curr(idx_closest);
                    else
                        peak_lin = 10.^(emp_val / 10);
                    end
                    idx_b = freqs >= tb(1) & freqs <= tb(2);
                    emp_ratio   = peak_lin / (sum(P_e_lin_curr(idx_b)) + eps);
                    emp_energy  = TSDTB_Core.calculatePSDEnergy(P_e_lin_curr, freqs, struct('centerFreq', tb));
                    emp_energy1 = TSDTB_Core.calculatePSDEnergy(P_e_log_curr, freqs, struct('centerFreq', tb));
                end
                
                if has_s
                    alpha_mask = locs_s >= freqs_alpha(1) & locs_s <= freqs_alpha(2);
                    if any(alpha_mask)
                        idx_list = find(alpha_mask);
                        [~, sub_idx] = max(pks_s(alpha_mask));
                        idx_s = idx_list(sub_idx);
                    else
                        [~, idx_s] = max(pks_s);
                    end
                    sim_freq  = locs_s(idx_s);
                    sim_val   = pks_s(idx_s);
                    sim_width = widths_s(idx_s);
                    sim_prom  = proms_s(idx_s);
                    sim_Q     = sim_freq / (sim_width + eps);
                    if ~has_e
                        tb = [max(sim_freq - sim_width/2 - 1, freqs(1)), ...
                              min(sim_freq + sim_width/2 + 1, freqs(end))];
                    end
                    if use_relative
                        [~, idx_closest_s] = min(abs(freqs - sim_freq));
                        peak_lin_s = P_s_lin_curr(idx_closest_s);
                    else
                        peak_lin_s = 10.^(sim_val / 10);
                    end
                    idx_b = freqs >= tb(1) & freqs <= tb(2);
                    sim_ratio   = peak_lin_s / (sum(P_s_lin_curr(idx_b)) + eps);
                    sim_energy  = TSDTB_Core.calculatePSDEnergy(P_s_lin_curr, freqs, struct('centerFreq', tb));
                    sim_energy1 = TSDTB_Core.calculatePSDEnergy(P_s_log_curr, freqs, struct('centerFreq', tb));
                end
                
                PENALTY = 1.0; REWARD = 0.0;
                if has_e && has_s
                    if abs(sim_freq - emp_freq) <= 2
                        res.peakFreq = TSDTB_Core.clip01(TSDTB_Core.relErr(sim_freq, emp_freq));
                        if use_relative
                            res.peakPower = TSDTB_Core.clip01(TSDTB_Core.relErr(peak_lin_s, peak_lin));
                        else
                            res.peakPower = TSDTB_Core.clip01(TSDTB_Core.relErr(10.^(sim_val/10), 10.^(emp_val/10)));
                        end
                        res.peakHeight  = TSDTB_Core.clip01(TSDTB_Core.relErr(sim_prom, emp_prom));
                        res.peakWidth   = TSDTB_Core.clip01(TSDTB_Core.relErr(sim_width, emp_width));
                        res.peakQ       = TSDTB_Core.clip01(TSDTB_Core.relErr(sim_Q, emp_Q));
                        res.peakRatio   = TSDTB_Core.clip01(TSDTB_Core.relErr(sim_ratio, emp_ratio));
                        res.peakEnergy  = TSDTB_Core.clip01(TSDTB_Core.relErr(sim_energy, emp_energy));
                        res.peakEnergy1 = TSDTB_Core.clip01(TSDTB_Core.relErr(sim_energy1, emp_energy1));
                    else
                        for k = 1:length(peak_fields), res.(peak_fields{k}) = PENALTY; end
                    end
                elseif xor(has_e, has_s)
                    for k = 1:length(peak_fields), res.(peak_fields{k}) = PENALTY; end
                else
                    for k = 1:length(peak_fields), res.(peak_fields{k}) = REWARD; end
                end
                
                for k = 1:numel(bandNames)
                    bName = bandNames{k};
                    band_struct = struct('centerFreq', bands.(bName));
                    val_raw_e  = TSDTB_Core.calculatePSDEnergy(P_e_lin_curr, freqs, band_struct);
                    val_raw_s  = TSDTB_Core.calculatePSDEnergy(P_s_lin_curr, freqs, band_struct);
                    val_norm_e = TSDTB_Core.calculatePSDEnergy(P_e_log_curr, freqs, band_struct);
                    val_norm_s = TSDTB_Core.calculatePSDEnergy(P_s_log_curr, freqs, band_struct);
                    
                    res.([bName, 'Energy'])  = TSDTB_Core.clip01(TSDTB_Core.relErr(val_raw_s, val_raw_e));
                    res.([bName, 'Energy1']) = TSDTB_Core.clip01(TSDTB_Core.relErr(val_norm_s, val_norm_e));
                end
                loop_res_arr(i) = res;
            end
            
            % Aperiodic Parameters (Offset & Exponent) & Aggregation
            if strcmpi(calc_mode, 'mean')
                Pe = P_emp_lin;
                Ps = P_sim_lin;
                if use_relative
                    Pe = (Pe - min(Pe)) / (max(Pe) - min(Pe) + eps);
                    Ps = (Ps - min(Ps)) / (max(Ps) - min(Ps) + eps);
                end
                emp_ap = TSDTB_Core.fitSingleFOOOF(freqs, Pe + eps);
                sim_ap = TSDTB_Core.fitSingleFOOOF(freqs, Ps + eps);
                
                offset_loss   = TSDTB_Core.clip01(TSDTB_Core.relErr(sim_ap.aperiodic_params(1), emp_ap.aperiodic_params(1)));
                exponent_loss = TSDTB_Core.clip01(TSDTB_Core.relErr(sim_ap.aperiodic_params(2), emp_ap.aperiodic_params(2)));
                results = loop_res_arr(1);
            else
                nCh = size(P_emp_lin_all, 1);
                emp_off = nan(nCh, 1); emp_exp = nan(nCh, 1);
                sim_off = nan(nCh, 1); sim_exp = nan(nCh, 1);
                for ch = 1:nCh
                    Pe = P_emp_lin_all(ch, :);
                    Ps = P_sim_lin_all(ch, :);
                    if use_relative
                        Pe = (Pe - min(Pe)) / (max(Pe) - min(Pe) + eps);
                        Ps = (Ps - min(Ps)) / (max(Ps) - min(Ps) + eps);
                    end
                    try
                        e_ap = TSDTB_Core.fitSingleFOOOF(freqs, Pe + eps);
                        s_ap = TSDTB_Core.fitSingleFOOOF(freqs, Ps + eps);
                        emp_off(ch) = e_ap.aperiodic_params(1);
                        emp_exp(ch) = e_ap.aperiodic_params(2);
                        sim_off(ch) = s_ap.aperiodic_params(1);
                        sim_exp(ch) = s_ap.aperiodic_params(2);
                    catch
                    end
                end
                offset_loss   = TSDTB_Core.clip01(median(TSDTB_Core.relErr(sim_off, emp_off), 'omitnan'));
                exponent_loss = TSDTB_Core.clip01(median(TSDTB_Core.relErr(sim_exp, emp_exp), 'omitnan'));
                results = struct();
                for k = 1:length(all_loss_fields)
                    fname = all_loss_fields{k};
                    results.(fname) = median([loop_res_arr.(fname)], 'omitnan');
                end
            end
            results.offset   = offset_loss;
            results.exponent = exponent_loss;
            
            % Merge spectral shape fitting metrics (aligned with psd_fitting.m)
            shape_fitting = TSDTB_Core.computePSDShapeMetrics(P_emp_log, P_sim_log);
            shape_fitting.lossPearson  = TSDTB_Core.clip01(1 - shape_fitting.Pearson);
            shape_fitting.lossSpearman = TSDTB_Core.clip01(1 - shape_fitting.Spearman);
            
            f_shape = fieldnames(shape_fitting);
            for k = 1:numel(f_shape)
                results.(f_shape{k}) = shape_fitting.(f_shape{k});
            end
        end
        
        function results = computePSDShapeMetrics(psd_emp, psd_sim)
            % Strictly aligned with psd_fitting.m
            [channelNum, ~] = size(psd_emp);
            pearson_ch  = zeros(channelNum, 1);
            spearman_ch = zeros(channelNum, 1);
            cosine_ch   = zeros(channelNum, 1);
            mse_norm_ch = zeros(channelNum, 1);
            diffabs_ch  = zeros(channelNum, 1);
            ks_ch       = zeros(channelNum, 1);
            emd_ch      = zeros(channelNum, 1);
            kl_ch       = zeros(channelNum, 1);
            bias_ch     = zeros(channelNum, 1);
            mse_raw_ch  = zeros(channelNum, 1);
            
            for c = 1:channelNum
                emp = psd_emp(c, :);
                sim = psd_sim(c, :);
                emp_norm = (emp - min(emp)) / (max(emp) - min(emp) + eps);
                sim_norm = (sim - min(sim)) / (max(sim) - min(sim) + eps);
                
                pearson_ch(c)  = corr(emp_norm(:), sim_norm(:));
                spearman_ch(c) = corr(emp_norm(:), sim_norm(:), 'Type', 'Spearman');
                cosine_ch(c)   = dot(emp_norm(:), sim_norm(:)) / (norm(emp_norm(:)) * norm(sim_norm(:)) + eps);
                mse_norm_ch(c) = immse(emp_norm, sim_norm);
                diffabs_ch(c)  = mean(abs(emp_norm - sim_norm));
                [~, ~, ks_ch(c)] = kstest2(emp_norm, sim_norm);
                emd_ch(c)      = sum(abs(sort(emp_norm) - sort(sim_norm))) / length(emp_norm);
                
                shift = 1e-10;
                emp_prob = (emp_norm + shift) / sum(emp_norm + shift);
                sim_prob = (sim_norm + shift) / sum(sim_norm + shift);
                kl1 = sum(emp_prob .* log(emp_prob ./ sim_prob));
                kl2 = sum(sim_prob .* log(sim_prob ./ emp_prob));
                kl_ch(c) = (kl1 + kl2) / 2;
                
                bias_ch(c)    = mean(sim - emp);
                mse_raw_ch(c) = mean((sim - emp).^2);
            end
            
            results.Pearson     = tanh(mean(atanh(pearson_ch)));
            results.Spearman    = tanh(mean(atanh(spearman_ch)));
            results.Cosine      = mean(cosine_ch);
            results.MSE_Norm    = mean(mse_norm_ch);
            results.MeanAbs     = mean(diffabs_ch);
            results.KS          = mean(ks_ch);
            results.Wasserstein = mean(emd_ch);
            results.KLsym       = mean(kl_ch);
            results.Bias_dB     = mean(abs(bias_ch));
            results.MSE_dB      = mean(mse_raw_ch);
            results.Bias_Raw    = mean(bias_ch);
            
            data_range = max(psd_emp(:)) - min(psd_emp(:));
            if data_range == 0, data_range = 1; end
            results.Bias_dB1 = TSDTB_Core.clip01(results.Bias_dB / data_range);
            results.MSE_dB1  = TSDTB_Core.clip01((sqrt(results.MSE_dB) / data_range)^2);
        end
        
        function res = computeMatrixFitting(m1, m2)
            % Lower triangular matrix comparison aligned with matrices_fitting.m
            nCols = size(m1, 2);
            Isubdiag = find(tril(true(nCols), -1));
            v1 = m1(Isubdiag); v2 = m2(Isubdiag);
            
            res.CCC      = 2 * corr(v1, v2) * std(v1) * std(v2) / (var(v1) + var(v2) + (mean(v1) - mean(v2))^2);
            res.Pearson  = corr(v1, v2, 'Type', 'Pearson');
            res.Spearman = corr(v1, v2, 'Type', 'Spearman');
            res.MSE      = immse(m1, m2);
            [~, ~, res.KS] = kstest2(v1, v2);
        end
        
        %% ================= 6. Utilities =================
        function err = relErr(sim, emp)
            err = abs(sim - emp) ./ (abs(emp) + eps);
        end
        
        function y = clip01(x)
            y = min(max(x, 0), 1);
            y(isnan(x)) = NaN;
        end
        
        function val = getSafeField(struc, field, default)
            if isfield(struc, field) && ~isempty(struc.(field))
                val = struc.(field);
            else
                val = default;
            end
        end
        
    end
end