function [ corr_para, mean_para ] = para_moments( j_mat, h_vec)

num_spin = size(j_mat, 1);
num_sample = 3 ^ num_spin; 
str_tri = dec2base(0: 3 ^ num_spin - 1, 3)';
ordered_sample = - 1 * (str_tri == '0') + 1 * (str_tri == '2');

j_mat = j_mat + diag(diag(j_mat)); 
ordered_energy = - (h_vec' * ordered_sample...
    + sum(j_mat * ordered_sample .* ordered_sample) / 2); 
ordered_exp = exp(- ordered_energy);
partition = sum(ordered_exp);
freq = ordered_exp / partition;

mean_para = sum(ordered_sample .* freq, 2); 

% per_batch = 1e5; 
mat1 = reshape(ordered_sample, [], num_spin, num_sample);
mat2 = reshape(ordered_sample, num_spin, [], num_sample);
premat = mat1 .* mat2;
freq3 = reshape(freq, 1, 1, num_sample);
corr_para = sum(freq3 .* premat, 3);


end