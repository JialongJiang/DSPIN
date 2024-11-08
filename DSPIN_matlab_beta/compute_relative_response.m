function relative_h = compute_relative_response(cur_h, if_control, batch_index)

    if all(if_control == 0)
    error('if_control contains all zeros');
    end

    unique_batches = unique(batch_index);
    num_batches = length(unique_batches);

    relative_h = zeros(size(cur_h));
    all_controls_mean = mean(cur_h(:, if_control), 2);
    
    for ii = 1: num_batches
        % Get the current batch index
        current_batch = unique_batches(ii);
        
        % Get the indices of samples in the current batch
        batch_samples_idx = find(batch_index == current_batch);
        
        % Get the control indices in the current batch
        batch_controls_idx = batch_samples_idx(if_control(batch_samples_idx));
        
        if ~isempty(batch_controls_idx)
            % Compute the mean of controls in the current batch
            batch_controls_mean = mean(cur_h(:, batch_controls_idx), 2);
        else
            % Use the mean of all controls if there are no controls in the current batch
            batch_controls_mean = all_controls_mean;
        end
        
        % Compute the relative h for samples in the current batch
        for idx = batch_samples_idx
            relative_h(:, idx) = cur_h(:, idx) - batch_controls_mean;
        end
    end



end

