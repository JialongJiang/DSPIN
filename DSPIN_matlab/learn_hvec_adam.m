function [ cur_h ] = learn_hvec_adam(corrs, means, train_dat)


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


rec_hgrad_norm = zeros(num_epoch, 1);
rec_hvec_all = zeros(num_spin, num_epoch); 
% rec_hgrad_all = zeros(num_spin, num_round, num_rec);


count = 1;
beta1 = 0.9; beta2 = 0.999; epsilon = 1e-6;
mhh = zeros(num_spin, 1); 
vhh = zeros(num_spin, 1); 
mhh_log = zeros(num_spin, num_epoch); 
vhh_log = zeros(num_spin, num_epoch); 


jj = 1;
while jj <= num_epoch

    samplingsz = round(min(jj * samplingsz_step, samplingsz_raw));  
    % samplingsz = samplingsz_raw;

    if num_spin <= spin_thres
        mean_para = para_mean(cur_j, cur_h);
    else
        mean_para = samp_mean(cur_j, cur_h, samplingsz, samplingmix, 1);
    end
    rec_hgrad = (mean_para - means);   

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
    
    
    rec_hvec_all(:, jj) = cur_h; 
    rec_hgrad_norm(jj) = sqrt(sum(rec_hgrad_full .^ 2, 1));

    mhh_log(:, jj) = mhh; 
    vhh_log(:, jj) = vhh; 
    
    
    if jj == list_step(count)
        save([task_name, '.mat'], 'list_step',...
        'rec_hvec_all', 'rec_hgrad_norm');

        fprintf('%d ', round(100 * count / num_rec));
        % plot(1: jj, rec_hgrad_norm(1: jj))
        % set(gca, 'YScale', 'log')
        % drawnow()
        count = count + 1;
    end

step_gap = 20;
if (stepsz > 1e-3) && (jj > step_gap) && (rec_hgrad_norm(jj) > 2 * rec_hgrad_norm(jj - step_gap))
    mhh = mhh_log(:, jj - step_gap); 
    vhh = vhh_log(:, jj - step_gap); 
    jj = jj - step_gap; 
    counter = counter - step_gap;
    stepsz = stepsz / 2;
    if count > num_rec
        count = count - 1; 
    end
else
    counter = counter + 1;
    jj = jj + 1;   
end

end

[val, pos] = min(rec_hgrad_norm); 

cur_h = rec_hvec_all(:, pos); 

end

