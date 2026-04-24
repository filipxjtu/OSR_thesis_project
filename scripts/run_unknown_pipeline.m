function unknown_data = run_unknown_pipeline(spec, n_per_class, dataset_seed)
    % RUN_UNKNOWN_PIPELINE
    % specifically targets the OSR Unknown
    % classes (10-13) by temporarily overriding the spec.class_ids.
    
    arguments
        spec (1,1) struct
        n_per_class (1,1) double {mustBeInteger, mustBePositive}
        dataset_seed (1,1) double {mustBeInteger, mustBeNonnegative}
    end
    
    project_root = fileparts(fileparts(mfilename('fullpath')));
    addpath(genpath(fullfile(project_root, 'matlab')));
    
    core.validate_spec_structure(spec);

    mode = 'eval';
    
    % Local Spec Setup
    spec_local = spec;
    spec_local.dataset_seed = int32(dataset_seed);
    
    % Override the target classes so the base generators process 10-13
    spec_local.class_ids = spec_local.unknown_class_ids; 
    
    version = spec_local.spec_version;
    
    % --- OPTIONAL: override SNR here when needed ---
    spec_local.snr_mode = "range";
    spec_local.snr_train_db = [-18 10];
    spec_local.snr_eval_db  = [-10 10];
        % OR
    % spec_local.snr_mode = "fixed";
    % spec_local.snr_fixed_db = -6;

    spec_local.snr_skew_gamma = 1.5;
    
    % Folders
    if ~exist('artifacts','dir'); mkdir('artifacts'); end
    output_dir = fullfile('artifacts','datasets','unknown');
    if ~exist(output_dir,'dir'); mkdir(output_dir); end
    
    fprintf('=== UNKNOWN PIPELINE START ===\n');
    fprintf('Seed: %d | n_per_class: %d | SNR Mode: %s\n', ...
            dataset_seed, n_per_class, spec_local.snr_mode);
            
    % Generate Clean Unknowns
    dataset = clean.generate_clean_dataset(n_per_class, spec_local);
    
    % Apply Impairments
    unknown_data = impaired.generate_impaired_dataset( ...
        dataset, spec_local, mode);

    unknown_data.meta.dataset_version = char("unknown_dataset_v2");
    
    % Save clean unknown dataset
    clean_filename = sprintf('clean_unk_dataset_%s_seed%d_n%d.mat', ...
                        version, dataset_seed, n_per_class);
    save(fullfile(output_dir, clean_filename), 'dataset', '-v7.3');

     % Save impaired unknown dataset
    filename = sprintf('unknown_dataset_%s_seed%d_n%d.mat', ...
                        version, dataset_seed, n_per_class);
    save(fullfile(output_dir, filename), 'unknown_data', '-v7.3');
    
    % report
    unknown.generate_stat_report_unknown(unknown_data);
    
    % Sanity Checks
    assert(isfield(unknown_data,'meta'), 'Missing meta.');
    assert(unknown_data.meta.dataset_seed == dataset_seed, ...
        'Seed mismatch.');
        
    fprintf('=== UNKNOWN PIPELINE COMPLETE ===\n');
end