function impaired_data = run_impaired_pipeline(spec, n_per_class, dataset_seed, mode)
%RUN_IMPAIRED_PIPELINE Master orchestrator for impaired dataset generation.
%
% Inputs:
%   spec          : canonical spec struct
%   n_per_class   : samples per class
%   dataset_seed  : deterministic seed
%   mode          : 'train' or 'eval'
%
% Output:
%   impaired      : impaired dataset struct

    arguments
        spec (1,1) struct
        n_per_class (1,1) double {mustBeInteger, mustBePositive}
        dataset_seed (1,1) double {mustBeInteger, mustBeNonnegative}
        mode (1,:) char
    end

    project_root = fileparts(fileparts(mfilename('fullpath')));
    addpath(genpath(fullfile(project_root, 'matlab')));

    mode = string(lower(mode));
    if mode ~= "train" && mode ~= "eval"
        error('run_impaired_pipeline:BadMode', ...
              'Mode must be "train" or "eval".');
    end

    % Inject seed
    spec_local = spec;
    spec_local.dataset_seed = int32(dataset_seed);
    version = spec_local.spec_version;

    % Ensure folders exist
    if ~exist('artifacts','dir'); mkdir('artifacts'); end
    if ~exist('reports','dir');   mkdir('reports');   end

    fprintf('=== IMPAIRED PIPELINE START ===\n');
    fprintf('Seed: %d | n_per_class: %d | Mode: %s\n', ...
            dataset_seed, n_per_class, mode);

    % Step 1: generate clean baseline
    clean_dataset = clean.generate_clean_dataset(n_per_class, spec_local);

    % Step 2: generate impaired dataset
    impaired_data = impaired.generate_impaired_dataset(clean_dataset, spec_local, mode);

    % Save artifact
    filename = sprintf('impaired_dataset_%s_seed%d_n%d_%s.mat', ...
                        version, dataset_seed, n_per_class, mode);
    output_dir = fullfile('artifacts','datasets','impaired');
    save(fullfile(output_dir, filename), 'impaired_data', '-v7.3');

    % Generate report
    impaired.generate_stat_report_impaired(impaired_data);

    fprintf('=== IMPAIRED PIPELINE COMPLETE ===\n');
end
