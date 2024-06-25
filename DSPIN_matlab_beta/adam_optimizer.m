function [vars, m, v] = adam_optimizer(vars, grads, m, v, t, learning_rate)

    beta1 = 0.9; beta2 = 0.999; epsilon = 1e-8;

    % Number of variables
    num_vars = length(vars);
    
    % Update each variable
    for i = 1:num_vars
        
        % Update biased first moment estimate
        m{i} = beta1 * m{i} + (1 - beta1) * grads{i};
        
        % Update biased second raw moment estimate
        v{i} = beta2 * v{i} + (1 - beta2) * (grads{i} .^ 2);
        
        % Compute bias-corrected first moment estimate
        m_hat = m{i} / (1 - beta1^t);
        
        % Compute bias-corrected second raw moment estimate
        v_hat = v{i} / (1 - beta2^t);
        
        % Update variable
        vars{i} = vars{i} - learning_rate * m_hat ./ (sqrt(v_hat) + epsilon);
    end
end

