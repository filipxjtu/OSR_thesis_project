function impaired_data = run_impaired_pipeline(spec, n_per_class, dataset_seed, mode)

    arguments
        spec (1,1) struct
        n_per_class (1,1) double {mustBeInteger, mustBePositive}
        dataset_seed (1,1) double {mustBeInteger, mustBeNonnegative}
        mode (1,1) string
        %xxx {mustBeInteger}
    end

    project_root = fileparts(fileparts(mfilename('fullpath')));
    addpath(genpath(fullfile(project_root, 'matlab')));

    core.validate_spec_structure(spec);

    mode = lower(mode);
    if mode ~= "train" && mode ~= "eval"
        error('run_impaired_pipeline:BadMode', ...
            'Mode must be "train" or "eval".');
    end
    % local spec
    spec_local = spec;
    spec_local.dataset_seed = int32(dataset_seed);
    version = spec_local.spec_version;

    % --- OPTIONAL: override SNR here when needed ---
    spec_local.snr_mode = "range";
    spec_local.snr_train_db = [5 15];
    spec_local.snr_eval_db  = [5 15];
    % OR
    %spec_local.snr_mode = "fixed";
    %spec_local.snr_fixed_db = xxx;
    
    if mode == "train"
        spec_local.snr_skew_gamma = 1.000000001;
    else
        spec_local.snr_skew_gamma = 1;
    end

    % folders
    if ~exist('artifacts','dir'); mkdir('artifacts'); end
    if ~exist('reports','dir');   mkdir('reports');   end

    output_dir = fullfile('artifacts','datasets','impaired');
    if ~exist(output_dir,'dir'); mkdir(output_dir); end

    fprintf('=== IMPAIRED PIPELINE START ===\n');
    fprintf('Seed: %d | n_per_class: %d | Mode: %s | SNR Mode: %s\n', ...
            dataset_seed, n_per_class, mode, spec_local.snr_mode);

    % Step 1: clean
    clean_dataset = clean.generate_clean_dataset(n_per_class, spec_local);

    % Step 2: impaired
    impaired_data = impaired.generate_impaired_dataset( ...
        clean_dataset, spec_local, mode);

    % save
    filename = sprintf('impaired_dataset_%s_seed%d_n%d_%s.mat', ...
                        version, dataset_seed, n_per_class, mode);

    save(fullfile(output_dir, filename), 'impaired_data', '-v7.3');

    % report
    impaired.generate_stat_report_impaired(impaired_data);

    % sanity
    assert(isfield(impaired_data,'meta'), 'Missing meta.');
    assert(impaired_data.meta.dataset_seed == dataset_seed, ...
        'Seed mismatch.');

    fprintf('=== IMPAIRED PIPELINE COMPLETE ===\n');
end