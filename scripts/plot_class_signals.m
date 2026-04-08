clear;
clc;

project_root = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(project_root,'matlab')));

spec = core.get_canonical_spec();
spec.dataset_seed = int32(123);

sample_idx = int32(0);
n = (0:spec.N-1);

out_dir = fullfile(project_root, 'artifacts', 'figs', ...
    sprintf('clean_seed%d', spec.dataset_seed));
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

for class_id = spec.class_ids

    [x,~,~] = clean.generate_clean_sample(class_id, sample_idx, spec);

    fig = figure('Visible','off', 'Position',[100 100 1600 900]);

    % --- Magnitude ---
    subplot(3,1,1)
    plot(n, abs(x), 'LineWidth', 1.5)
    grid on
    ylabel('|x[n]|')
    title(sprintf('Magnitude — Class %d', class_id))

    % --- Real & Imag ---
    subplot(3,1,2)
    plot(n, real(x), 'LineWidth', 1.2)
    hold on
    plot(n, imag(x), '--', 'LineWidth', 1.2)
    grid on
    ylabel('Amplitude')
    legend('Real','Imag')

    % --- Phase ---
    subplot(3,1,3)
    plot(n, unwrap(angle(x)), 'LineWidth', 1.2)
    grid on
    xlabel('Sample index')
    ylabel('Phase (rad)')

    filename = fullfile(out_dir, sprintf('class_%02d.png', class_id));
    exportgraphics(fig, filename, 'Resolution', 300);

    close(fig);

end