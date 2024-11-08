data_dir = '../'; 
task_name = 'thomsonlab_signaling';

sampling_sz = 1e6; 

task_name_res = [data_dir, task_name, '/dspin/'];
[ ~, ~ ] = mkdir(task_name_res); 


load([task_name_res, 'raw_data.mat'])
cdata = raw_data; 

num_spin = size(cdata{1}{1}, 1);
num_samp = size(cdata, 2);
rec_all_corr = zeros(num_spin, num_spin, num_samp);
rec_all_mean = zeros(num_spin, num_samp);

for ii = 1: num_samp
    rec_all_corr(:, :, ii) = cdata{ii}{1};
    rec_all_mean(:, ii)  = cdata{ii}{2};
end

cur_j = zeros(num_spin); 
cur_h = zeros(num_spin, num_samp); 
train_dat = struct('cur_j', cur_j, 'cur_h', cur_h, 'epoch', 400,'spin_thres', 16,...
    'stepsz', 0.02, 'dropout', 0, 'counter', 1,...
    'samplingsz', sampling_sz, 'samplingmix', 1e3, 'rec_gap', 5,...
    'lam_l2h', 0.005, 'lam_l1j', 0.01);
train_dat.task_name = [task_name_res, 'train_net']; 

if num_spin <= train_dat.spin_thres
    train_dat.stepsize = 0.2; 
    train_dat.lam_l2h = 0; 
    train_dat.lam_l1j = 0; 
end

if num_samp > 3
    % parpool(min(256, num_samp))
end

[cur_j, cur_h] = learn_jmat_adam(rec_all_corr, rec_all_mean, train_dat);

save([task_name_res, '/network.mat'], 'cur_h', 'cur_j')
