
data_dir = '../thomsonlab_signaling/'; 
task_name = 'thomsonlab_signaling';
data_name = 'signaling_raw_data.mat';
training_algorithm = 'pseudo_likelihood';
directed = false; 

save_path = [data_dir, task_name, '/'];
[ ~, ~ ] = mkdir(save_path); 

load([data_dir, data_name])
cdata = raw_data; 

num_samp = size(cdata, 2);

if strcmp(training_algorithm, 'likelihood')

    num_spin = size(cdata{1}{1}, 1);
    rec_all_corr = zeros(num_spin, num_spin, num_samp);
    rec_all_mean = zeros(num_spin, num_samp);
    for ii = 1:num_samp
        rec_all_corr(:, :, ii) = cdata{ii}{1};
        rec_all_mean(:, ii)  = cdata{ii}{2};
    end

elseif strcmp(training_algorithm, 'pseudo_likelihood')

    num_spin = size(cdata{1}, 2);

    for ii = 1: num_samp
        cdata{ii} = cdata{ii}'; %(1: 5, :)
    end

end

cur_j = zeros(num_spin); 
cur_h = zeros(num_spin, num_samp); 
train_dat = struct('cur_j', cur_j, 'cur_h', cur_h, 'epoch', 400, 'spin_thres', 16,...
    'stepsz', 0.05, 'rec_gap', 20, 'method', training_algorithm, 'directed', directed, 'save_path', save_path);
train_dat.if_control = if_control;
train_dat.batch_index = batch_index;

if strcmp(training_algorithm, 'likelihood')

    train_dat.sampling_sz = 1e6;
    train_dat.samplingmix = 1e3;

    if strcmp(likelihood_type, 'mcmc')
        train_dat.stepsz = 0.02; 
    elseif strcmp(likelihood_type, 'exact')
        train_dat.stepsz = 0.2; 
    end

elseif strcmp(training_algorithm, 'pseudo_likelihood')

    train_dat.stepsz = 0.05; 

end

train_dat.lam_l1j = 0.01; 
train_dat.lam_l2j = 0; 
train_dat.lam_l1h = 0;
train_dat.lam_l2h = 0;

train_dat.lam_l1h_control = 0;


if num_samp > 3
    parpool(min(12, num_samp))
end

[cur_j, cur_h] = learn_jmat_adam(cdata, train_dat);

save([save_path, 'network.mat'], 'cur_h', 'cur_j')
