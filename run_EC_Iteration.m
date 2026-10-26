function [EC_opt, iter_results] = run_EC_Iteration(params_set)
    % run_EC_Iteration Executes the EC iteration
    
    fprintf('\n--- [Stage 1] Executing EC Iteration ---\n');
    iterNum     = params_set.IterParams.iterNum;
    initial_lr  = params_set.IterParams.initial_lr;
    decay_rate  = params_set.IterParams.decay_rate;
    min_lr      = params_set.IterParams.min_lr;
    max_lr      = params_set.IterParams.max_lr;
    inc_factor  = params_set.IterParams.lr_increase_factor;
    dec_factor  = params_set.IterParams.lr_decrease_factor;
    
    SC_new      = params_set.SC_emp_norm;
    G_e         = TSDTB_Core.getSafeField(params_set, 'G_e', 1.0);
    target_FC   = TSDTB_Core.getSafeField(params_set, 'EC_fc', params_set.FC_emp_norm);
    
    prev_corr   = 0;
    current_lr  = initial_lr;
    best_loss   = inf;
    best_iter   = 1;
    EC_opt      = SC_new - diag(diag(SC_new));
    
    iter_results = struct('G_e', {}, 'RE_mean', {}, 'time_decayed_lr', {}, ...
                          'LearningRate', {}, 'FC_corr', {}, 'FC_ksDis', {}, ...
                          'FC_mse', {}, 'EC_iter', {}, 'FC_iter', {});
    
    for iter = 1:iterNum
        EC = SC_new - diag(diag(SC_new));
        
        % Simulate dynamics with BOLD enabled
        [~, ~, simFC, bold_results] = TSDTB_Core.simulateDynamics(params_set, EC, G_e, true);
        
        current_corr = bold_results.Pearson;
        current_ks   = bold_results.KS;
        current_mse  = bold_results.MSE;
        firing_rate  = bold_results.RE_mean;
        
        % Time-decay learning rate schedule
        time_decayed_lr = initial_lr / (1 + decay_rate * iter);
        
        if iter > 1
            if current_corr > prev_corr
                adjusted_lr = time_decayed_lr * (1 + inc_factor);
            else
                adjusted_lr = time_decayed_lr * (1 + dec_factor);
            end
            current_lr = min(max(adjusted_lr, min_lr), max_lr);
        end
        prev_corr = current_corr;
        
        fprintf('  (Iter %02d/%02d) FiringRate: %.4f | DecayedLR: %.4f | LR: %.4f | FC Corr: %.4f | KS Dist: %.4f\n', ...
            iter, iterNum, firing_rate, time_decayed_lr, current_lr, current_corr, current_ks);
        
        % Record iteration metrics
        iter_results(iter).G_e             = G_e;
        iter_results(iter).RE_mean         = firing_rate;
        iter_results(iter).time_decayed_lr = time_decayed_lr;
        iter_results(iter).LearningRate    = current_lr;
        iter_results(iter).FC_corr         = current_corr;
        iter_results(iter).FC_ksDis        = current_ks;
        iter_results(iter).FC_mse          = current_mse;
        iter_results(iter).EC_iter         = EC;
        iter_results(iter).FC_iter         = simFC;
        
        % Track optimal EC minimizing (1 - FC_corr + FC_ksDis)
        current_loss = 1 - current_corr + current_ks;
        if current_loss < best_loss
            best_loss = current_loss;
            best_iter = iter;
            EC_opt    = EC;
        end
        
        % Gradient update of SC_new
        SC_new = SC_new + current_lr .* (target_FC - simFC);
    end
    
    fprintf('--- Stage 1 Complete | Optimal Iter: %d (FC Corr: %.4f, KS: %.4f, Loss: %.4f) ---\n', ...
        best_iter, iter_results(best_iter).FC_corr, iter_results(best_iter).FC_ksDis, best_loss);
end