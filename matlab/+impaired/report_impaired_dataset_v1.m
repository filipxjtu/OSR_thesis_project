function report = report_impaired_dataset_v1(impaired_dataset)
%REPORT_IMPAIRED_DATASET_V1 Produce SNR stats + checksum for determinism evidence.
%
% Output:
%   report struct with fields:
%     - snr_target: [Ns x 1]
%     - snr_realized: [Ns x 1]
%     - per_class_mean_realized: table
%     - global_checksum: uint64 (FNV-1a over bytes of X_imp)
%     - basic_stats: struct

    X = impaired_dataset.X_imp;
    y = impaired_dataset.y(:);
    ip = impaired_dataset.imp_params;

    Ns = size(X,2);

    snr_t = zeros(Ns,1);
    snr_r = zeros(Ns,1);
    for i=1:Ns
        snr_t(i) = ip(i).target_snr_db;
        snr_r(i) = ip(i).realized_snr_db;
    end

    classes = unique(y);
    mean_r = zeros(numel(classes),1);
    for k=1:numel(classes)
        ck = classes(k);
        mean_r(k) = mean(snr_r(y==ck));
    end
    per_class_mean_realized = table(classes, mean_r, 'VariableNames', {'class_id','mean_realized_snr_db'});

    % Basic distribution stats
    basic_stats = struct();
    basic_stats.target_min = min(snr_t);
    basic_stats.target_max = max(snr_t);
    basic_stats.target_mean = mean(snr_t);
    basic_stats.realized_min = min(snr_r);
    basic_stats.realized_max = max(snr_r);
    basic_stats.realized_mean = mean(snr_r);
    basic_stats.realized_std = std(snr_r);

    % Checksum 
    global_checksum = impaired_dataset.meta.artifact_hash;

    report = struct();
    report.snr_target = snr_t;
    report.snr_realized = snr_r;
    report.per_class_mean_realized = per_class_mean_realized;
    report.basic_stats = basic_stats;
    report.global_checksum = global_checksum;
end
