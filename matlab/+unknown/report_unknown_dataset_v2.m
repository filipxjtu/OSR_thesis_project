function report = report_unknown_dataset_v2(unknown_dataset)
    X = unknown_dataset.X_imp;
    y = unknown_dataset.y(:);
    ip = unknown_dataset.imp_params;
    
    % strict validation
    assert(isfield(unknown_dataset,'meta'), 'Missing meta.');
    assert(numel(ip) == size(X,2), 'imp_params size mismatch.');
    
    Ns = size(X,2);
    snr_t = zeros(Ns,1);
    snr_r = zeros(Ns,1);
    
    for i = 1:Ns
        t = ip(i).target_snr_db;
        r = ip(i).realized_snr_db;
        assert(isfinite(t), 'Invalid target_snr_db at %d', i);
        assert(isfinite(r), 'Invalid realized_snr_db at %d', i);
        snr_t(i) = t;
        snr_r(i) = r;
    end
    
    % per-class stats 
    classes = unique(y);
    mean_r = zeros(numel(classes),1);
    for k = 1:numel(classes)
        ck = classes(k);
        idx = (y == ck);
        mean_r(k) = mean(snr_r(idx));
    end
    
    per_class_mean_realized = table( ...
        classes, mean_r, ...
        'VariableNames', {'class_id','mean_realized_snr_db'} ...
    );
    
    % basic stats
    basic_stats = struct();
    basic_stats.target_min = min(snr_t);
    basic_stats.target_max = max(snr_t);
    basic_stats.target_mean = mean(snr_t);
    basic_stats.realized_min = min(snr_r);
    basic_stats.realized_max = max(snr_r);
    basic_stats.realized_mean = mean(snr_r);
    basic_stats.realized_std = std(snr_r);
    
    % checksum
    global_checksum = unknown_dataset.meta.artifact_hash;
    
    % assemble
    report = struct();
    report.snr_target = snr_t;
    report.snr_realized = snr_r;
    report.per_class_mean_realized = per_class_mean_realized;
    report.basic_stats = basic_stats;
    report.global_checksum = global_checksum;
end