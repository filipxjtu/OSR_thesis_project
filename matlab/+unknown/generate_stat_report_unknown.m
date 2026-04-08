function generate_stat_report_unknown(unknown_dataset)
% generates report for the unknown data

    arguments
        unknown_dataset (1,1) struct
    end
    
    meta = unknown_dataset.meta;
    seed = meta.dataset_seed;
    mode = meta.mode;
    snr_mode = meta.snr_mode;
    N = meta.N;
    Ns = meta.Ns;
    n_per_class = meta.n_per_class;
    spec_version = meta.spec_version;
    
    % report data
    report = unknown.report_unknown_dataset_v2(unknown_dataset);
    
    % file
    filename = sprintf( ...
        'unknown_dataset_%s_seed%d_n%d_statistical_report.md', ...
        spec_version, seed, n_per_class);
        
    intended_dir = fullfile('reports','statistical');
    if ~exist(intended_dir,'dir'); mkdir(intended_dir); end
    
    fid = fopen(fullfile(intended_dir, filename), 'w');
    assert(fid ~= -1, 'Failed to open report file.');
    
    % header
    fprintf(fid, '# Unknown Dataset Artifact Report (v2)\n');
    fprintf(fid, 'Version: unknown_dataset_%s  \n', spec_version);
    fprintf(fid, 'Mode: %s (Strictly Evaluation) \n', mode);
    fprintf(fid, 'SNR Mode: %s  \n', snr_mode);
    fprintf(fid, 'Artifact: unknown_dataset_%s_seed%d_n%d.mat  \n\n', ...
        spec_version, seed, n_per_class);
        
    % specs
    fprintf(fid, '---\n\n');
    fprintf(fid, '## 1. Dataset Specifications\n\n');
    fprintf(fid, '| Parameter | Value |\n');
    fprintf(fid, '|-----------|-------|\n');
    fprintf(fid, '| dataset_seed | %d |\n', seed);
    fprintf(fid, '| N | %d |\n', N);
    fprintf(fid, '| N_samples | %d |\n', Ns);
    fprintf(fid, '| SNR Mode | %s |\n\n', snr_mode);
    
    % checksum
    fprintf(fid, '---\n\n');
    fprintf(fid, '## 2. Integrity Evidence\n\n');
    fprintf(fid, 'Global checksum (simple64_checksum):\n\n');
    fprintf(fid, '```\n%u\n```\n\n', report.global_checksum);
    
    % SNR stats
    bs = report.basic_stats;
    fprintf(fid, '---\n\n');
    fprintf(fid, '## 3. SNR Statistics (dB)\n\n');
    fprintf(fid, '| Metric | Target | Realized |\n');
    fprintf(fid, '|--------|--------|----------|\n');
    fprintf(fid, '| Min | %.2f | %.2f |\n', bs.target_min, bs.realized_min);
    fprintf(fid, '| Max | %.2f | %.2f |\n', bs.target_max, bs.realized_max);
    fprintf(fid, '| Mean | %.2f | %.2f |\n', bs.target_mean, bs.realized_mean);
    fprintf(fid, '| Std | - | %.2f |\n\n', bs.realized_std);
    
    % per-class
    fprintf(fid, '---\n\n');
    fprintf(fid, '## 4. Per-Class Realized SNR (dB)\n\n');
    fprintf(fid, '| Class ID | Target Anomaly Type | Mean Realized SNR |\n');
    fprintf(fid, '|----------|---------------------|-------------------|\n');
    pc = report.per_class_mean_realized;
    
    % Optional mapping for nice markdown tables
    anomaly_names = containers.Map({10, 11, 12, 13}, ...
        {'Moving-Band Noise', 'Intermittent OFDM', 'RVGPO', 'Triangular FM'});
        
    for i = 1:height(pc)
        cid = pc.class_id(i);
        aname = 'Unknown';
        if isKey(anomaly_names, cid)
            aname = anomaly_names(cid);
        end
        fprintf(fid, '| %d | %s | %.2f |\n', ...
            cid, aname, pc.mean_realized_snr_db(i));
    end
    
    fclose(fid);
    fprintf('Unknown dataset report generated: %s\n', filename);
end