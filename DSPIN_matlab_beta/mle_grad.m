function [j_grad, h_grad] = mle_grad(j_mat, h_vec, cur_data)


    cur_corr = cur_data{1};
    cur_mean = cur_data{2};

    [corr_para, mean_para] = para_moments(j_mat, h_vec);

    j_grad = cur_corr - corr_para;
    h_grad = cur_mean - mean_para;  
    
end

