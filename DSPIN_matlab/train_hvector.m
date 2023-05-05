%{
data_dir = '../20221003_genome_perturb_seq_k562_essential';
task_name = 'genome_perturb_seq_k562_essential_all';
cur_ind = 5; 
sampling_sz = 1e7; 
%}

task_name_res = [data_dir, '/dspin/', task_name, '/'];
% [ ~, ~ ] = mkdir(task_name_res); 

load([data_dir, '/dspin/', task_name, '_raw.mat'])
cdata = raw_data; 
num_spin = size(cdata{1}{1}, 1);
rec_corr = cdata{cur_ind}{1};
rec_mean  = cdata{cur_ind}{2};

load([task_name_res, '/network.mat'])


cur_h = zeros(num_spin, 1); 
train_dat = struct('cur_j', cur_j, 'cur_h', cur_h, 'epoch', 500,'spin_thres', 16,...
    'decayp', 0.3, 'stepsz', 0.05, 'dropout', 0, 'counter', 1,...
    'samplingsz', sampling_sz, 'samplingmix', 1e3, 'rec_gap', 20,...
    'lam_l2h', 0.005, 'lam_l1j', 0.01);

[ ~, ~ ] = mkdir([task_name_res, '/train_each/']); 
[ ~, ~ ] = mkdir([task_name_res, '/train_each/log']); 
train_dat.task_name = [task_name_res, '/train_each/log/log_', num2str(cur_ind)]; 

cur_h = learn_hvec_adam(rec_corr, rec_mean, train_dat);

[ ~, ~ ] = mkdir([task_name_res, '/train_each/res']); 
save([task_name_res, '/train_each/res/res_', num2str(cur_ind), '.mat'], 'cur_h')
