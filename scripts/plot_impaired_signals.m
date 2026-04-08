clear;
clc;

project_root = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(project_root,'matlab')));

spec = core.get_canonical_spec();
spec.dataset_seed = int32(17);

spec.snr_mode = "range";
spec.snr_train_db = [-12 -2];
spec.snr_eval_db  = [-2 8];

sample_idx = int32(0);
mode = "train";
n = (0:spec.N-1);

out_dir = fullfile(project_root, 'artifacts', 'figs', ...
    sprintf('impaired_seed%d', spec.dataset_seed));
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

for class_id = spec.class_ids

    [x_clean,~,~] = clean.generate_clean_sample(class_id, sample_idx, spec);
    [x_imp, ip]   = impaired.apply_impairment(x_clean, sample_idx, spec, mode);

    % --- invisible, large figure ---
    fig = figure('Visible','off', 'Position',[100 100 1600 900]);

    % --- Magnitude ---
    subplot(3,1,1)
    plot(n, abs(x_imp), 'LineWidth', 1.5)
    grid on
    ylabel('|x[n]|')
    title(sprintf('Magnitude — Class %d | SNR %.2f / %.2f dB', ...
        class_id, ip.target_snr_db, ip.realized_snr_db))

    % --- Real & Imag ---
    subplot(3,1,2)
    plot(n, real(x_imp), 'LineWidth', 1.2)
    hold on
    plot(n, imag(x_imp), '--', 'LineWidth', 1.2)
    grid on
    ylabel('Amplitude')
    legend('Real','Imag')

    % --- Phase ---
    subplot(3,1,3)
    plot(n, unwrap(angle(x_imp)), 'LineWidth', 1.2)
    grid on
    xlabel('Sample index')
    ylabel('Phase (rad)')

    % --- high-resolution export ---
    filename = fullfile(out_dir, sprintf('class_%02d.png', class_id));
    exportgraphics(fig, filename, 'Resolution', 300);

    close(fig);

end