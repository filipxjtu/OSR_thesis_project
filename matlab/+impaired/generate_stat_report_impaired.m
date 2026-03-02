function generate_stat_report_impaired(impaired_dataset)
%GENERATE_IMPAIRED_REPORT Generate markdown report from impaired dataset struct.
%
% Input:
%   impaired_dataset : struct from generate_impaired_dataset (saved .mat)

    arguments
        impaired_dataset (1,1) struct
    end

    % Get metadata from the dataset itself
    meta = impaired_dataset.meta;
    seed = meta.dataset_seed;
    mode = meta.mode;
    N = meta.N;  % Add this line
    Ns = meta.Ns;
    
    
    % Get report data using existing reporter
    report = impaired.report_impaired_dataset_v1(impaired_dataset);
    
    % Create filename
    filename = sprintf('impaired_dataset_v1_seed%d_%s_report.md', seed, mode);
    intended_dir = fullfile('reports','statistical');
    fid = fopen(fullfile(intended_dir, filename), 'w');
    
    % Write header
    fprintf(fid, '# Impaired Dataset Artifact Report\n');
    fprintf(fid, 'Version: impaired_dataset_v1  \n');
    fprintf(fid, 'Mode: %s  \n', mode);
    fprintf(fid, 'Artifact: impaired_dataset_v1_seed%d_%s.mat  \n\n', seed, mode);
    
    % Dataset specs
    fprintf(fid, '---\n\n');
    fprintf(fid, '## 1. Dataset Specifications\n\n');
    fprintf(fid, '| Parameter | Value |\n');
    fprintf(fid, '|-----------|-------|\n');
    fprintf(fid, '| dataset_seed | %d |\n', seed);
    fprintf(fid, '| N (signal length) | %d samples |\n', N);
    fprintf(fid, '| N_samples | %d |\n', Ns);
    fprintf(fid, '| Mode | %s |\n\n', mode);
    
    % Global checksum
    fprintf(fid, '---\n\n');
    fprintf(fid, '## 2. Integrity Evidence\n\n');
    fprintf(fid, 'Global checksum (FNV-1a 64-bit):\n\n');
    fprintf(fid, '```\n%u\n```\n\n', report.global_checksum);
    
    % SNR statistics
    fprintf(fid, '---\n\n');
    fprintf(fid, '## 3. SNR Statistics (dB)\n\n');
    fprintf(fid, '| Metric | Target | Realized |\n');
    fprintf(fid, '|--------|--------|----------|\n');
    fprintf(fid, '| Min | %.2f | %.2f |\n', report.basic_stats.target_min, report.basic_stats.realized_min);
    fprintf(fid, '| Max | %.2f | %.2f |\n', report.basic_stats.target_max, report.basic_stats.realized_max);
    fprintf(fid, '| Mean | %.2f | %.2f |\n', report.basic_stats.target_mean, report.basic_stats.realized_mean);
    fprintf(fid, '| Std | - | %.2f |\n\n', report.basic_stats.realized_std);
    
    % Per-class realized SNR
    fprintf(fid, '---\n\n');
    fprintf(fid, '## 4. Per-Class Realized SNR (dB)\n\n');
    fprintf(fid, '| Class ID | Mean Realized SNR |\n');
    fprintf(fid, '|----------|-------------------|\n');
    
    pc = report.per_class_mean_realized;
    for i = 1:height(pc)
        fprintf(fid, '| %d | %.2f |\n', pc.class_id(i), pc.mean_realized_snr_db(i));
    end
    
    % Optional: first few samples for inspection
    fprintf(fid, '\n---\n\n');
    fprintf(fid, '## 5. First 5 Samples Preview\n\n');
    fprintf(fid, '| Sample Index | Target SNR | Realized SNR |\n');
    fprintf(fid, '|--------------|------------|--------------|\n');
    
    for i = 1:min(5, length(report.snr_target))
        fprintf(fid, '| %d | %.2f | %.2f |\n', i-1, report.snr_target(i), report.snr_realized(i));
    end
    
    fclose(fid);
    fprintf('Impaired dataset report generated: %s\n', filename);
end