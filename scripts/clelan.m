clear;
clc;
close all;

% --- 1. Project Setup ---
project_root = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(project_root,'matlab')));

spec = core.get_canonical_spec();
spec.dataset_seed = int32(18);
spec.snr_mode = "range";
spec.snr_train_db = [-5 5];

fig_name = sprintf('thesis_plots_%d', spec.dataset_seed);
out_dir = fullfile(project_root, 'artifacts', 'figs',fig_name);
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

% --- 2. Physics & Time Vector ---
% You have 1024 samples at 10 MHz
fs = double(10e6); 
N = double(spec.N);
t_us = double((0:N-1)) / fs * 1e6; % Time vector in microseconds

sample_idx = int32(0);
mode = "train";

% --- 3. Figure Styling Settings ---
fontName = 'Times New Roman';
fontSize = 10;
cleanColor = [0 0 1]; % Black for clean signal
impColor = [0.8 0.2 0.2 0.7]; % Transparent red for impaired signal

for class_id = spec.class_ids
    
    % Generate data
    [x_clean, ~, ~] = clean.generate_clean_sample(class_id, sample_idx, spec);
    [x_imp, ip]     = impaired.apply_impairment(x_clean, class_id, sample_idx, spec, mode);

    % Create a wide, thesis-friendly figure
    fig = figure('Visible','off', 'Color', 'w', 'Position', [100 100 1200 800]);

    % --- Subplot 1: Magnitude (Envelope) ---
    subplot(3, 1, 1);
    plot(t_us, abs(x_clean), 'Color', cleanColor, 'LineWidth', 1.5); hold on;
    plot(t_us, abs(x_imp), 'Color', impColor, 'LineWidth', 0.8);
    grid on;
    xlabel('Time (\mus)', 'FontName', fontName, 'FontSize', fontSize);
    ylabel('Magnitude |x(t)|', 'FontName', fontName, 'FontSize', fontSize);
    title(sprintf('Signal Envelope: Class %d (SNR: %.2f dB)', class_id, ip.realized_snr_db), ...
        'FontName', fontName, 'FontSize', fontSize + 2, 'FontWeight', 'bold');
    legend('Clean', 'Impaired', 'Location', 'best', 'FontName', fontName);
    set(gca, 'FontName', fontName, 'FontSize', fontSize);

    % --- Subplot 2: Unwrapped Phase ---
    subplot(3, 1, 2);
    plot(t_us, unwrap(angle(x_clean)), 'Color', cleanColor, 'LineWidth', 1.5); hold on;
    plot(t_us, unwrap(angle(x_imp)), 'Color', impColor, 'LineWidth', 0.8);
    grid on;
    xlabel('Time (\mus)', 'FontName', fontName, 'FontSize', fontSize);
    ylabel('Unwrapped Phase (rad)', 'FontName', fontName, 'FontSize', fontSize);
    legend('Clean', 'Impaired', 'Location', 'best', 'FontName', fontName);
    set(gca, 'FontName', fontName, 'FontSize', fontSize);

    % --- Subplot 3: I/Q Components (Impaired Only for clarity) ---
    subplot(3, 1, 3);
    plot(t_us, real(x_imp), 'b', 'LineWidth', 0.8); hold on;
    plot(t_us, imag(x_imp), 'Color', [0.9 0.5 0], 'LineStyle', '--', 'LineWidth', 0.8);
    grid on;
    xlabel('Time (\mus)', 'FontName', fontName, 'FontSize', fontSize);
    ylabel('Amplitude', 'FontName', fontName, 'FontSize', fontSize);
    title('I/Q Components (Impaired Signal)', 'FontName', fontName, 'FontSize', fontSize, 'FontWeight', 'bold');
    legend('In-Phase (Real)', 'Quadrature (Imag)', 'Location', 'best', 'FontName', fontName);
    set(gca, 'FontName', fontName, 'FontSize', fontSize);

    % --- High-Resolution Vector Export ---
    % Saving as PDF ensures perfect quality in LaTeX/Word
    filename = fullfile(out_dir, sprintf('thesis_class_%02d.pdf', class_id));
    exportgraphics(fig, filename, 'ContentType', 'vector');
    
    close(fig);
end