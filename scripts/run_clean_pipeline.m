function dataset = run_clean_pipeline(spec, n_per_class, dataset_seed)
%RUN_CLEAN_PIPELINE Master orchestrator for clean dataset generation.

    arguments
        spec (1,1) struct
        n_per_class (1,1) double {mustBeInteger, mustBePositive}
        dataset_seed (1,1) double {mustBeInteger, mustBeNonnegative}
    end

    project_root = fileparts(fileparts(mfilename('fullpath')));
    addpath(genpath(fullfile(project_root, 'matlab')));

    core.validate_spec_structure(spec);

    % local spec
    spec_local = spec;

    % override seed
    spec_local.dataset_seed = int32(dataset_seed);

    version = spec_local.spec_version;

    if ~exist('artifacts','dir'); mkdir('artifacts'); end
    if ~exist('reports','dir');   mkdir('reports');   end

    output_dir = fullfile('artifacts', 'datasets','clean');
    if ~exist(output_dir,'dir'); mkdir(output_dir); end

    fprintf('=== CLEAN PIPELINE START ===\n');
    fprintf('Seed: %d | n_per_class: %d\n', dataset_seed, n_per_class);

    % generate dataset
    dataset = clean.generate_clean_dataset(n_per_class, spec_local);

    % save
    filename = sprintf( ...
        'clean_dataset_%s_seed%d_n%d.mat', ...
        version, dataset_seed, n_per_class);

    save(fullfile(output_dir, filename), 'dataset', '-v7.3');

    % report
    clean.generate_stat_report_clean(dataset);
    fprintf('=== CLEAN PIPELINE COMPLETE ===\n');
end