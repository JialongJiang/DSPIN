function [j_grad, h_grad] = pseudo_grad(j_mat, h_vec, cur_state, directed)
    
    num_spin = size(j_mat, 1);

    j_grad = zeros(num_spin, num_spin);
    h_grad = zeros(num_spin, 1);
    j_mat_off = j_mat - diag(diag(j_mat)); 
    effective_h = j_mat_off * cur_state + h_vec;
    
    for ii = 1: num_spin
        j_sub = j_mat(ii, ii);
        h_sub = effective_h(ii, :);

        term1 = exp(j_sub + h_sub);
        term2 = exp(j_sub - h_sub);

        j_sub_grad = cur_state(ii, :) .^ 2 - (term1 + term2) ./ (term1 + term2 + 1);
        h_eff_grad = cur_state(ii, :) - (term1 - term2) ./ (term1 + term2 + 1);

        j_off_sub_grad = h_eff_grad .* cur_state;
        j_grad(ii, :) = mean(j_off_sub_grad, 2);
        j_grad(ii, ii) = mean(j_sub_grad);

        h_grad(ii) = mean(h_eff_grad);
        
        if ~ directed
            j_grad = (j_grad + j_grad') / 2;
        end
        
    end

    h_grad = - h_grad; 
    j_grad = - j_grad; 
    
end

