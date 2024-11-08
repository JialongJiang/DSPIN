
data_dir = '../thomsonlab_signaling/'; 
task_name = 'thomsonlab_signaling';
data_name = 'signaling_raw_data.mat';
training_algorithm = 'likelihood'; % or 'pseudolikelihood';
likelihood_type = 'mcmc'; % or 'exact';

save_path = [data_dir, task_name, '/'];
[ ~, ~ ] = mkdir(save_path); 

load([data_dir, data_name])
cdata = raw_data; 

num_samp = size(cdata, 2);

if training_algorithm == 'likelihood'

    num_spin = size(cdata{1}{1}, 1);
    rec_all_corr = zeros(num_spin, num_spin, num_samp);
    rec_all_mean = zeros(num_spin, num_samp);
    for ii = 1: num_samp
        rec_all_corr(:, :, ii) = cdata{ii}{1};
        rec_all_mean(:, ii)  = cdata{ii}{2};
    end

elseif training_algorithm == 'pseudolikelihood'

    num_spin = size(cdata{1}, 1);
    for ii = 1: num_samp
        temp_data = cdata{ii}';
        % cdata{ii} = temp_data(1: 80, :);
        cdata{ii} = temp_data;
    end

end


cur_j = zeros(num_spin); 
cur_h = zeros(num_spin, num_samp); 
train_dat = struct('cur_j', cur_j, 'cur_h', cur_h, 'epoch', 300, 'spin_thres', 16,...
    'stepsz', 0.02, 'rec_gap', 5, 'save_path', save_path); 

if training_algorithm == 'likelihood'

    train_dat.sampling_sz = 1e6;
    train_dat.samplingmix = 1e3;

    if likelihood_type == 'mcmc'
        train_dat.stezsz = 0.02; 
    elseif likelihood_type == 'exact'
        train_dat.stepsize = 0.2; 
    end

elseif training_algorithm == 'pseudolikelihood'

    train_dat.stepsz = 0.05; 

end

train_dat.lam_l1j = 0.01; 
train_dat.lam_l2j = 0; 
train_dat.lam_l1h = 0;
train_dat.lam_l2h = 0.005;

train_dat.lam_l2h_perturb = 0; 
train_dat.lam_l1h_control = 0;


if num_samp > 3
    % parpool(min(256, num_samp))
end

[cur_j, cur_h] = learn_jmat_adam(rec_all_corr, rec_all_mean, train_dat);

save([save_path, 'network.mat'], 'cur_h', 'cur_j')
