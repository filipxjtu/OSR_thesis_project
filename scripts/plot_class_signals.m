% plot_clean_signals.m
clear;
clc;

project_root = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(project_root,'matlab')));

spec = core.get_canonical_spec();
spec.dataset_seed = int32(123);

sample_idx = int32(0);

for class_id = 0:6

    [x,~,~] = clean.generate_clean_sample(class_id, sample_idx, spec);

    figure

    plot(0:spec.N-1, x, 'LineWidth', 1)
    grid on

    xlabel('Sample index')
    ylabel('Amplitude')

    title(sprintf('Clean Signal — Class %d', class_id))

    xlim([0 spec.N])

end