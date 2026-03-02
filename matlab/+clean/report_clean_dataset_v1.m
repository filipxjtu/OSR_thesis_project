function report = report_clean_dataset_v1(clean_dataset)
%REPORT_CLEAN_DATASET_V1 Produce stats + checksum for clean dataset.
%
% Output:
%   report struct with fields:
%     - per_class_stats: table (mean, std, rms per class)
%     - global_checksum: double (sum(abs(X_clean(:))))
%     - basic_stats: struct

    X = clean_dataset.X_clean;
    y = clean_dataset.y(:);
    
    Ns = size(X,2);
    classes = unique(y);
    
    % Per-class statistics
    class_ids = zeros(numel(classes),1);
    mean_vals = zeros(numel(classes),1);
    std_vals = zeros(numel(classes),1);
    rms_vals = zeros(numel(classes),1);
    
    for k = 1:numel(classes)
        ck = classes(k);
        idx = (y == ck);
        Xk = X(:, idx);
        
        class_ids(k) = ck;
        mean_vals(k) = mean(Xk(:));
        std_vals(k) = std(Xk(:));
        rms_vals(k) = sqrt(mean(Xk(:).^2));
    end
    
    per_class_stats = table(class_ids, mean_vals, std_vals, rms_vals, ...
        'VariableNames', {'class_id', 'mean', 'std', 'rms'});
    
    % Basic stats
    basic_stats = struct();
    basic_stats.global_mean = mean(X(:));
    basic_stats.global_std = std(X(:));
    basic_stats.global_rms = sqrt(mean(X(:).^2));
    
    % Checksum (same as your existing meta.checksum)
    global_checksum = sum(abs(X(:)));
    
    % Assemble report
    report = struct();
    report.per_class_stats = per_class_stats;
    report.basic_stats = basic_stats;
    report.global_checksum = global_checksum;
    report.N = size(X,1);
    report.N_samples = Ns;
end