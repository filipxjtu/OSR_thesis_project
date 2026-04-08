function generate_stat_report_impaired(impaired_dataset)

    arguments
        impaired_dataset (1,1) struct
    end

    %validation
    assert(isfield(impaired_dataset,'meta'), 'Missing meta.');

    meta = impaired_dataset.meta;

    required_meta = ["dataset_seed","mode","Ns","N","n_per_class","spec_version","snr_mode"];
    for f = required_meta
        assert(isfield(meta, f), 'Missing meta field: %s', f);
    end

    seed = meta.dataset_seed;
    mode = meta.mode;
    snr_mode = meta.snr_mode;
    N = meta.N;
    Ns = meta.Ns;
    n_per_class = meta.n_per_class;
    spec_version = meta.spec_version;

    % report data
    report = impaired.report_impaired_dataset_v2(impaired_dataset);

    % file
    filename = sprintf( ...
        'impaired_dataset_%s_seed%d_n%d_%s_report.md', ...
        spec_version, seed, n_per_class, mode);

    intended_dir = fullfile('reports','statistical');
    if ~exist(intended_dir,'dir'); mkdir(intended_dir); end

    fid = fopen(fullfile(intended_dir, filename), 'w');
    assert(fid ~= -1, 'Failed to open report file.');

    % header
    fprintf(fid, '# Impaired Dataset Artifact Report (v2)\n');
    fprintf(fid, 'Version: impaired_dataset_%s  \n', spec_version);
    fprintf(fid, 'Mode: %s  \n', mode);
    fprintf(fid, 'SNR Mode: %s  \n', snr_mode);
    fprintf(fid, 'Artifact: impaired_dataset_%s_seed%d_n%d_%s.mat  \n\n', ...
        spec_version, seed, n_per_class, mode);

    % specs
    fprintf(fid, '---\n\n');
    fprintf(fid, '## 1. Dataset Specifications\n\n');
    fprintf(fid, '| Parameter | Value |\n');
    fprintf(fid, '|-----------|-------|\n');
    fprintf(fid, '| dataset_seed | %d |\n', seed);
    fprintf(fid, '| N | %d |\n', N);
    fprintf(fid, '| N_samples | %d |\n', Ns);
    fprintf(fid, '| Mode | %s |\n', mode);
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
    fprintf(fid, '| Class ID | Mean Realized SNR |\n');
    fprintf(fid, '|----------|-------------------|\n');

    pc = report.per_class_mean_realized;
    for i = 1:height(pc)
        fprintf(fid, '| %d | %.2f |\n', ...
            pc.class_id(i), pc.mean_realized_snr_db(i));
    end

    % preview
    fprintf(fid, '\n---\n\n');
    fprintf(fid, '## 5. First 5 Samples Preview\n\n');
    fprintf(fid, '| Sample Index | Target SNR | Realized SNR |\n');
    fprintf(fid, '|--------------|------------|--------------|\n');

    for i = 1:min(5, length(report.snr_target))
        fprintf(fid, '| %d | %.2f | %.2f |\n', ...
            i-1, report.snr_target(i), report.snr_realized(i));
    end

    fclose(fid);

    fprintf('Impaired dataset report generated: %s\n', filename);
end