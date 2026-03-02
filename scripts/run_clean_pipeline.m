function dataset = run_clean_pipeline(spec, n_per_class, dataset_seed)
%RUN_CLEAN_PIPELINE Master orchestrator for clean dataset generation.
%
% Inputs:
%   spec          : canonical spec struct (without dataset_seed required)
%   n_per_class   : number of samples per class
%   dataset_seed  : deterministic seed for this dataset instance
%
% Output:
%   dataset       : clean dataset struct


    arguments
        spec (1,1) struct
        n_per_class (1,1) double {mustBeInteger, mustBePositive}
        dataset_seed (1,1) double {mustBeInteger, mustBeNonnegative}
    end

    project_root = fileparts(fileparts(mfilename('fullpath')));
    addpath(genpath(fullfile(project_root, 'matlab')));

    % Inject seed (do NOT mutate original spec)
    spec_local = spec;
    spec_local.dataset_seed = int32(dataset_seed);

    % Ensure folders exist
    if ~exist('artifacts','dir'); mkdir('artifacts'); end
    if ~exist('reports','dir');   mkdir('reports');   end

    fprintf('=== CLEAN PIPELINE START ===\n');
    fprintf('Seed: %d | n_per_class: %d\n', dataset_seed, n_per_class);

    % Generate dataset
    dataset = clean.generate_clean_dataset(n_per_class, spec_local);

    % Save artifact
    filename = sprintf('clean_dataset_v1_seed%d.mat', dataset_seed);
    output_dir = fullfile('artifacts', 'datasets','clean');
    save(fullfile(output_dir, filename), 'dataset', '-v7.3');

    % Generate report
    clean.generate_stat_report_clean(dataset);

    fprintf('=== CLEAN PIPELINE COMPLETE ===\n');
end
