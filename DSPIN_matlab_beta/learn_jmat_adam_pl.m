function [ cur_j, cur_h ] = learn_jmat_adam_pl(state_list, train_dat)

num_round = size(state_list, 2); 
num_spin = size(state_list{1}, 1);

state_size = zeros(num_round, 1); 
for ii = 1: num_round
    state_size(ii) = size(state_list{ii}, 2);
end
samp_weight = sqrt(state_size);
samp_weight = samp_weight / sum(samp_weight);

num_epoch = train_dat.epoch;
stepsz = train_dat.stepsz; 
save_path = train_dat.save_path; 

rec_gap = train_dat.rec_gap;
list_step = fliplr(num_epoch: - rec_gap: 1); 
num_rec = length(list_step);

cur_j = train_dat.cur_j; cur_h = train_dat.cur_h; 

rec_jgrad = zeros(num_spin, num_spin, num_round);
rec_hgrad = zeros(num_spin, num_round);

rec_hgrad_norm = zeros(num_round, num_epoch);
rec_hvec_all = zeros(num_spin, num_round, num_epoch); 

rec_jgrad_norm = zeros(num_round, num_epoch); 
rec_jgrad_sum_norm = ones(num_epoch, 1) * 1e6; 
rec_jmat_all = zeros(num_spin, num_spin, num_epoch); 

mm = {zeros(num_spin), zeros(num_spin, num_round)};
vv = {zeros(num_spin), zeros(num_spin, num_round)};

mm_log = cell(num_epoch, 1); 
vv_log = cell(num_epoch, 1); 

backtrap_tol = 4; 
cur_backtrap = 0;

counter = 1;
plot_counter = 1; 
while counter <= num_epoch
        parfor kk = 1: num_round
            [corr_grad, mean_grad] = pseudo_grad(cur_j, cur_h(:, kk), state_list{kk});

            rec_jgrad(:, :, kk) = - corr_grad;
            rec_hgrad(:, kk) = - mean_grad;   
        end

        
    rec_jgrad_full = rec_jgrad; 
    rec_jgrad = sum(rec_jgrad .* reshape(samp_weight, 1, 1, []), 3); 
    
    if isfield(train_dat, 'lam_l1j')
        rec_jgrad = rec_jgrad + train_dat.lam_l1j * min(max(cur_j / 1e-4, -1), 1);
    end
    
    if isfield(train_dat, 'lam_l2h')
        rec_hgrad = rec_hgrad + train_dat.lam_l2h * cur_h; end
    
    
    if isfield(train_dat, 'if_control')
        relative_h = compute_relative_response(cur_h, train_dat.if_control, train_dat.batch_index); 
    end

    if isfield(train_dat, 'lam_l2h_perturb')
        rec_hgrad = rec_hgrad + train_dat.lam_l2h_perturb .* (relative_h - train_dat.perturb_matrix);
    end
    
    if isfield(train_dat, 'lam_l1h_perturb')
        rec_hgrad = rec_hgrad + train_dat.lam_l1h_perturb .* min(max(relative_h / 1e-4, -1), 1);
    end
   
    [vars, mm, vv] = adam_optimizer({cur_j, cur_h},...
        {rec_jgrad, rec_hgrad}, mm, vv, counter, stepsz);
    
    cur_j = vars{1}; 
    cur_h = vars{2}; 
       
    
    rec_hvec_all(:, :, counter) = cur_h; 
    rec_jmat_all(:, :, counter) = cur_j; 

    rec_hgrad_norm(:, counter) = sqrt(sum(rec_hgrad .^ 2, 1));
    rec_jgrad_norm(:, counter) = sqrt(sum(rec_jgrad_full .^ 2, [1, 2]));
    % rec_jgrad_sum_norm(jj) = sqrt(sum(sum(rec_jgrad_full, 3) .^ 2, [1, 2]));
    rec_jgrad_sum_norm(counter) = sqrt(sum(rec_jgrad .^ 2, [1, 2]));
    
    mm_log{counter} = mm; 
    vv_log{counter} = vv; 
        
    
    if counter == list_step(plot_counter)
        
        fprintf('%d ', round(100 * plot_counter / num_rec));
        
        if plot_counter > 1
        subplot(2, 2, 1)
        boxplot(rec_jgrad_norm(:, list_step(1: plot_counter)), list_step(1: plot_counter), 'Whisker', 1)
        set(gca, 'YScale', 'log')
        end
        
        subplot(2, 2, 2)
        plot(1: counter, rec_jgrad_sum_norm(1: counter), 'LineWidth', 4)
        set(gca, 'YScale', 'log')
        
        if plot_counter > 1
        subplot(2, 2, 3)
        boxplot(rec_hgrad_norm(:, list_step(1: plot_counter)), list_step(1: plot_counter), 'Whisker', 1)
        set(gca, 'YScale', 'log')
        end
        
        num_plot = min(50, num_spin); 
        subplot(2, 2, 4)
        imagesc(cur_j(1: num_plot, 1: num_plot))
        colormap(flipud(redblue))
        caxis([- 0.25, 0.25])
        
        colorbar()
        drawnow()
        saveas(gcf, [save_path, '/log.fig'])
        save([save_path, '/network_mlog.mat'], 'rec_jmat_all', 'rec_hvec_all', 'rec_jgrad_sum_norm')
        plot_counter = plot_counter + 1;
    end

step_gap = 40;
if (counter > step_gap) && (rec_jgrad_sum_norm(counter) > 2 * rec_jgrad_sum_norm(counter - step_gap))
    mm = mm_log{counter - step_gap};
    vv = vv_log{counter - step_gap};
    cur_j = rec_jmat_all(:, :, counter - step_gap);
    cur_h = rec_hvec_all(:, :, counter - step_gap);

    counter = counter - step_gap;
    stepsz = stepsz / 2;

    disp(['backtrack ', num2str(stepsz)])
    cur_backtrap = cur_backtrap + 1;
    if cur_backtrap > backtrap_tol
        disp('backtrack tolerence reached')
        break;
    end
else
    counter = counter + 1;
end


end

[val, pos] = min(rec_jgrad_sum_norm); 

cur_h = rec_hvec_all(:, :, pos); 
cur_j = rec_jmat_all(:, :, pos); 

end

