% plot_impaired_signals.m
clear;
clc;

project_root = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(project_root,'matlab')));

spec = core.get_canonical_spec();
spec.dataset_seed = int32(123);

sample_idx = int32(0);
mode = "train";

for class_id = 0:6

    % generate clean signal
    [x_clean,~,~] = clean.generate_clean_sample(class_id, sample_idx, spec);

    % apply impairment
    [x_imp, ~] = impaired.apply_impairment(x_clean, sample_idx, spec, mode);

    % plot impaired signal
    figure
    plot(0:spec.N-1, x_imp, 'LineWidth',1)
    grid on

    xlabel('Sample index')
    ylabel('Amplitude')

    title(sprintf('Impaired Signal — Class %d', class_id))

    xlim([0 spec.N])

end