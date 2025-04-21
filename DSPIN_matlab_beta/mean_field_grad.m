function [j_grad, h_grad] = mean_field_grad(j_mat, h_vec, cur_state, directed)

    num_spin = size(j_mat, 1);

    j_grad = zeros(num_spin, num_spin);
    h_grad = zeros(num_spin, 1);
    j_mat_off = j_mat - diag(diag(j_mat)); 
    j_mat_diag = diag(j_mat); 

    mean_expr = (exp(j_mat_diag + h_vec) - exp(j_mat_diag - h_vec)) ./ (exp(j_mat_diag + h_vec) + exp(j_mat_diag - h_vec) + 1); 
    
    max_iter = 100; 
    % save_iter = zeros(max_iter, num_spin); 
    
    for kk = 1: max_iter
        % effective_h = j_mat * mean_expr + h_vec; 
        % expr_new = (exp(effective_h) - exp(- effective_h)) ./ (exp(effective_h) + exp(- effective_h) + 1); 
        effective_h = j_mat_off * mean_expr + h_vec; 
        expr_new = (exp(j_mat_diag + effective_h) - exp(j_mat_diag - effective_h)) ./ (exp(j_mat_diag + effective_h) + exp(j_mat_diag - effective_h) + 1); 
        update_norm = norm(mean_expr - expr_new) / (norm(mean_expr) + 1e-8); 

        mean_expr = expr_new; 
    
        if update_norm < 0.01
            break
        end
    end
    
    if update_norm > 0.1
        disp('Warning: mean field not converging')
    end
    
    effective_h = j_mat_off * mean_expr + h_vec;
    
    for ii = 1: num_spin
        j_sub = j_mat(ii, ii);
        h_sub = effective_h(ii, :);

        term1 = exp(j_sub + h_sub);
        term2 = exp(j_sub - h_sub);

        j_sub_grad = cur_state(ii, :) .^ 2 - (term1 + term2) ./ (term1 + term2 + 1);
        h_eff_grad = cur_state(ii, :) - (term1 - term2) ./ (term1 + term2 + 1);

        j_off_sub_grad = h_eff_grad .* mean_expr;
        j_grad(ii, :) = mean(j_off_sub_grad, 2);
        j_grad(ii, ii) = mean(j_sub_grad);

        h_grad(ii) = mean(h_eff_grad);
        
        if ~ directed
            j_grad = (j_grad + j_grad') / 2;
        end
        
    end

    h_grad = - h_grad; 
    j_grad = - j_grad; 

    % j_grad = j_grad - diag(diag(j_grad)); 
    
end   
