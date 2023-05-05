data_dir = '../data/'; 
task_name = 'thomsonlab_signaling';

task_name_res = [data_dir, task_name, '/dspin/'];

load([task_name_res, 'data_raw.mat'])
cdata = raw_data; 
num_spin = size(cdata{1}{1}, 1);
num_samp = size(cdata, 2); 

load([task_name_res, '/network.mat'])

num_epoch = 500; 
h_all = zeros(num_spin, num_samp); 
all_hgrad_norm = zeros(num_samp, num_epoch); 

for cur_ind = 1: num_samp
    try
    load([task_name_res, '/train_each/log/log_', num2str(cur_ind), '.mat'])
    all_hgrad_norm(cur_ind, :) = rec_hgrad_norm; 
    
    load([task_name_res, '/train_each/res/res_', num2str(cur_ind), '.mat'])
    h_all(:, cur_ind) = cur_h; 
    catch
    fprintf(int2str(cur_ind))
    fprintf(' ')
    end
      
    
end

sz_gap = 20;

boxplot(all_hgrad_norm(:, 1: sz_gap: num_epoch), 1: sz_gap: num_epoch, 'Whisker', 1)
set(gca, 'YScale', 'log')
        
cur_h = h_all; 

       
save([task_name_res, '/all_data.mat'], 'cur_h', 'cur_j')

