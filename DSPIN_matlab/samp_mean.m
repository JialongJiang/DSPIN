function [ mean_para ] = samp_mean( j_mat, h_vec, sample_size, mixing_time, samp_gap)
%Sample configurations of spin
%   This function make spin samples at given parameter. The J and H shall be
%   scaled by the inverse tempreture beta. 

per_batch = 1e6; 
rec_mean = 0; 

%% Network parameters
num_spin = size(j_mat, 1);

%% Sampling parameters
beta = 1;
batch_count = 1;
rec_sample = ones(num_spin, per_batch);

%% Gibbs sampling
% cur_spin = - ones(num_spin, 1);
cur_spin = randi([0, 2], num_spin, 1) - 1; 
tot_sampling = mixing_time + sample_size * samp_gap - mod(mixing_time, samp_gap);
  
rand_ind = randi([1, num_spin], tot_sampling, 1);

rand_flip = randi([0, 1], tot_sampling, 1); 
rand_prob = rand(tot_sampling, 1);
for ii = 1: tot_sampling
    cur_ind = rand_ind(ii);
    if cur_spin(cur_ind) == 0
        if rand_flip(ii) == 0
            new_spin = 1;           
        else
            new_spin = - 1;
        end
        diff_energy = - j_mat(cur_ind, cur_ind) - new_spin * ...
            (j_mat(cur_ind, :) * cur_spin + h_vec(cur_ind)); 
        accept_prob = min(1, exp(- diff_energy * beta));
    else
        if rand_flip(ii) == 0
            accept_prob = 0; 
        else
            diff_energy = cur_spin(cur_ind) * ... 
                (j_mat(cur_ind, :) * cur_spin + h_vec(cur_ind)); 
            accept_prob = min(1, exp(- diff_energy * beta));
        end
    end
    

    if rand_prob(ii) < accept_prob 
        if cur_spin(cur_ind) == 0
            cur_spin(cur_ind) = new_spin; 
        else
            cur_spin(cur_ind) = 0; 
        end
    end    
    if ii > mixing_time
        if rem(ii, samp_gap) == 0
            rec_sample(:, batch_count) = cur_spin;
            batch_count = batch_count + 1; 
            
            if batch_count == (per_batch + 1)
                batch_count = 1; 
                rec_mean = rec_mean + sum(rec_sample, 2); 
            end
        
        end
    end
end


%{
[samp_unique, ia, ic] = unique(rec_sample', 'rows'); 
[freq, ordered_sample] = para_distribution_tri(j_mat, h_vec);
freq_new = diff([0, find(diff(sort(ic)))', numel(ic)]) / sample_size;
scatter(freq_new, freq)
%}

if batch_count ~= 1
    cur_sample = rec_sample(:, 1: batch_count - 1); 
    rec_mean = rec_mean + sum(cur_sample, 2); 
end

mean_para = rec_mean / sample_size; 

end