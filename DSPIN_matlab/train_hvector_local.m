data_dir = '../data/'; 
task_name = 'thomsonlab_signaling';

sampling_sz = 5e6; % 1e7; 

task_name_res = [data_dir, task_name, '/dspin/'];
[ ~, ~ ] = mkdir(task_name_res); 


load([task_name_res, 'data_raw.mat'])
cdata = raw_data; 

load([task_name_res, 'network.mat'])

num_samp = size(raw_data, 2); 
rec_allh = zeros(num_spin, num_samp); 

[ ~, ~ ] = mkdir([task_name_res, '/train_each/']); 
[ ~, ~ ] = mkdir([task_name_res, '/train_each/log']); 
[ ~, ~ ] = mkdir([task_name_res, '/train_each/res']); 


for cur_ind = 1: num_samp

    num_spin = size(cdata{1}{1}, 1);
    rec_corr = cdata{cur_ind}{1};
    rec_mean  = cdata{cur_ind}{2};

    cur_h = zeros(num_spin, 1); 
    train_dat = struct('cur_j', cur_j, 'cur_h', cur_h, 'epoch', 500,'spin_thres', 16,...
        'stepsz', 0.05, 'dropout', 0, 'counter', 1,...
        'samplingsz', sampling_sz, 'samplingmix', 1e3, 'rec_gap', 20,...
        'lam_l2h', 0.005, 'lam_l1j', 0.01);
    if num_spin <= train_dat.spin_thres
        train_dat.stepsize = 0.2; 
        train_dat.lam_l2h = 0; 
        train_dat.lam_l1j = 0; 
    end
    train_dat.task_name = [task_name_res, '/train_each/log/log_', num2str(cur_ind)]; 

    cur_h = learn_hvec_adam(rec_corr, rec_mean, train_dat);
    
    parsave([task_name_res, '/train_each/res/res_', num2str(cur_ind), '.mat'], cur_h)
    rec_allh(:, cur_ind) = cur_h;
end

function parsave(fname, cur_h)
  save(fname, 'cur_h')
end

