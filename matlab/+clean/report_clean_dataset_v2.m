function report = report_clean_dataset_v2(clean_dataset)
%REPORT_CLEAN_DATASET_V1 Produce stats + checksum for clean dataset.
%
% Output:
%   report struct with fields:
%     - per_class_stats: table (mean, std, rms per class)
%     - global_checksum: double (sum(abs(X_clean(:))))
%     - basic_stats: struct

    assert(isfield(clean_dataset,'X_clean'), 'Missing X_clean.');
    assert(isfield(clean_dataset,'y'), 'Missing y.');
    assert(isfield(clean_dataset,'meta'), 'Missing meta.');

    X = clean_dataset.X_clean;
    y = clean_dataset.y(:);
    meta = clean_dataset.meta;
    
    Ns = size(X,2);
    classes = meta.class_set(:);
    
    % Per-class statistics
    nC = numel(classes);
    class_ids = zeros(nC,1);
    mean_vals = zeros(nC,1);
    std_vals = zeros(nC,1);
    rms_vals = zeros(nC,1);
    
    for k = 1:nC
        ck = classes(k);
        idx = (y == ck);
        assert(any(idx), 'Class %d has no samples.', ck);
        Xk = X(:, idx);
        
        class_ids(k) = ck;
        xk_mag = abs(Xk(:));
        mean_vals(k) = mean(xk_mag);
        std_vals(k) = std(xk_mag);
        rms_vals(k) = sqrt(mean(xk_mag.^2));
    end
    
    per_class_stats = table(class_ids, mean_vals, std_vals, rms_vals, ...
        'VariableNames', {'class_id', 'mean', 'std', 'rms'});
    
    % Basic stats
    X_mag = abs(X(:));
    basic_stats = struct();
    basic_stats.global_mean = mean(X_mag);
    basic_stats.global_std = std(X_mag);
    basic_stats.global_rms = sqrt(mean(X_mag.^2));
    
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