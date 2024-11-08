function [ cur_j, cur_h ] = learn_jmat_adam(corrs, means, train_dat)

num_round = size(corrs, 3); 
num_spin = size(corrs, 1);

num_epoch = train_dat.epoch;
stepsz = train_dat.stepsz; 
counter = train_dat.counter;
task_name = train_dat.task_name; 
samplingsz_raw = train_dat.samplingsz; samplingmix = train_dat.samplingmix;
rec_gap = train_dat.rec_gap; 
spin_thres = train_dat.spin_thres; 
samplingsz_step = samplingsz_raw / num_epoch * 2; 

list_step = fliplr(num_epoch: - rec_gap: 1); 
num_rec = length(list_step);

cur_j = train_dat.cur_j; cur_h = train_dat.cur_h; 

rec_jgrad = zeros(num_spin, num_spin, num_round);
rec_hgrad = zeros(num_spin, num_round);

rec_hgrad_norm = zeros(num_round, num_epoch);
rec_hvec_all = zeros(num_spin, num_round, num_epoch); 
% rec_hgrad_all = zeros(num_spin, num_round, num_rec);

rec_jgrad_norm = zeros(num_round, num_epoch); 
rec_jgrad_sum_norm = zeros(num_epoch, 1); 
rec_jmat_all = zeros(num_spin, num_spin, num_epoch); 
% rec_jgrad_all = zeros(num_spin, num_spin, num_round, num_rec); 

count = 1;
beta1 = 0.9; beta2 = 0.999; epsilon = 1e-6;
mjj = zeros(num_spin); 
vjj = zeros(num_spin); 
mhh = zeros(num_spin, num_round); 
vhh = zeros(num_spin, num_round); 
mjj_log = zeros(num_spin, num_spin, num_epoch); 
vjj_log = zeros(num_spin, num_spin, num_epoch); 
mhh_log = zeros(num_spin, num_round, num_epoch); 
vhh_log = zeros(num_spin, num_round, num_epoch); 


jj = 1;
while jj <= num_epoch

    samplingsz = round(min(jj * samplingsz_step, samplingsz_raw));  
    % samplingsz = samplingsz_raw;
    if num_round > 3
        parfor kk = 1: num_round
            if num_spin <= spin_thres
                [corr_para, mean_para] = para_moments(cur_j, cur_h(:, kk));
            else
                [corr_para, mean_para] = samp_moments(cur_j, cur_h(:, kk), samplingsz, samplingmix, 1);
            end
            rec_jgrad(:, :, kk) = (corr_para - corrs(:, :, kk));
            rec_hgrad(:, kk) = (mean_para - means(:, kk));   
        end
    else
        for kk = 1: num_round
            if num_spin <= spin_thres
                [corr_para, mean_para] = para_moments(cur_j, cur_h(:, kk));
            else
                [corr_para, mean_para] = samp_moments(cur_j, cur_h(:, kk), samplingsz, samplingmix, 1);
            end
            rec_jgrad(:, :, kk) = (corr_para - corrs(:, :, kk));
            rec_hgrad(:, kk) = (mean_para - means(:, kk));   
        end
    end

    if isfield(train_dat, 'lam_l1j')
        rec_jgrad = rec_jgrad + train_dat.lam_l1j * ...
            sign(wthresh(cur_j, 's', 1e-3));
    end

    rec_jgrad_full = rec_jgrad; 
    rec_jgrad = sum(rec_jgrad, 3) / num_round;
    mjj = beta1 .* mjj + (1 - beta1) .* rec_jgrad;
    vjj = beta2 .* vjj + (1 - beta2) .* (rec_jgrad .^ 2);

    mHatjj = mjj ./ (1 - beta1^counter);
    vHatjj = vjj ./ (1 - beta2^counter);

    vfStepjj = stepsz .* mHatjj ./ (sqrt(vHatjj) + epsilon);
    cur_j = cur_j - vfStepjj;
    
    if isfield(train_dat, 'lam_l2h')
        rec_hgrad = rec_hgrad + train_dat.lam_l2h * cur_h;
    end
    
    rec_hgrad_full = rec_hgrad; 
    mhh = beta1 .* mhh + (1 - beta1) .* rec_hgrad;
    vhh = beta2 .* vhh + (1 - beta2) .* (rec_hgrad .^ 2);

    mHathh = mhh ./ (1 - beta1^counter);
    vHathh = vhh ./ (1 - beta2^counter);

    vfStephh = stepsz .* mHathh ./ (sqrt(vHathh) + epsilon);
    cur_h = cur_h - vfStephh; 
    
    
    rec_hvec_all(:, :, jj) = cur_h; 
    rec_jmat_all(:, :, jj) = cur_j; 

    rec_hgrad_norm(:, jj) = sqrt(sum(rec_hgrad_full .^ 2, 1));
    rec_jgrad_norm(:, jj) = sqrt(sum(rec_jgrad_full .^ 2, [1, 2]));
    rec_jgrad_sum_norm(jj) = sqrt(sum(sum(rec_jgrad_full, 3) .^ 2, [1, 2]));
    
    mjj_log(:, :, jj) = mjj; 
    vjj_log(:, :, jj) = vjj;
    mhh_log(:, :, jj) = mhh; 
    vhh_log(:, :, jj) = vhh; 
        
    
    if jj == list_step(count)
 
        save([task_name, '.mat'], 'count', 'list_step',...
            'rec_hvec_all', 'rec_jmat_all', 'rec_hgrad_norm', ...
            'rec_jgrad_norm', 'rec_jgrad_sum_norm');
        
        fprintf('%d ', round(100 * count / num_rec));
        
        if count > 1
        subplot(2, 2, 1)
        boxplot(rec_jgrad_norm(:, list_step(1: count)), list_step(1: count), 'Whisker', 1)
        set(gca, 'YScale', 'log')
        end
        
        subplot(2, 2, 2)
        plot(1: jj, rec_jgrad_sum_norm(1: jj), 'LineWidth', 4)
        set(gca, 'YScale', 'log')
        
        if count > 1
        subplot(2, 2, 3)
        boxplot(rec_hgrad_norm(:, list_step(1: count)), list_step(1: count), 'Whisker', 1)
        set(gca, 'YScale', 'log')
        end

        subplot(2, 2, 4)
        imagesc(cur_j)
        colormap(flipud(redblue))
        caxis([- 1, 1])
        
        colorbar()
        drawnow()
        saveas(gcf, [task_name, '_log.fig'])
        count = count + 1;
    end

step_gap = 20;
if (jj > step_gap) && (rec_jgrad_sum_norm(jj) > 2 * rec_jgrad_sum_norm(jj - step_gap))
    mjj = mjj_log(:, :, jj - step_gap); 
    vjj = vjj_log(:, :, jj - step_gap); 
    mhh = mhh_log(:, :, jj - step_gap); 
    vhh = vhh_log(:, :, jj - step_gap); 
    jj = jj - step_gap; 
    counter = counter - step_gap;
    stepsz = stepsz / 2;
else
    counter = counter + 1;
    jj = jj + 1;   
end

end

[val, pos] = min(rec_jgrad_sum_norm); 

cur_h = rec_hvec_all(:, :, pos); 
cur_j = rec_jmat_all(:, :, pos); 

end

