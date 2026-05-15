function unknown_data = run_pro_test_pipeline(spec, n_per_class, dataset_seed, mode, xxx)
    % run_pro_test_pipeline
    % Generates an unknown dataset for OSR. The `mode` argument selects
    % which subset of unknown class ids to use (proxy vs test) and writes
    % the result with a `_proxy.mat` or `_test.mat` filename suffix.
    %
    % Spec contract additions:
    %   spec.unknown_proxy_class_ids   - integer vector of class IDs to
    %                                    generate when mode == "proxy"
    %   spec.unknown_test_class_ids    - integer vector of class IDs to
    %                                    generate when mode == "test"
    %
    % Existing fields (`spec.unknown_class_ids`) remain valid and are not
    % consulted by this function. Define the proxy/test fields once in
    % your spec setup, e.g.:
    %   spec.unknown_proxy_class_ids = [10, 11];  
    %   spec.unknown_test_class_ids  = [12, 13];   
    %
    % Filenames written:
    %   artifacts/datasets/unknown/unknown_dataset_{ver}_seed{s}_n{n}_proxy.mat
    %   artifacts/datasets/unknown/unknown_dataset_{ver}_seed{s}_n{n}_test.mat

    arguments
        spec        (1,1) struct
        n_per_class (1,1) double {mustBeInteger, mustBePositive}
        dataset_seed (1,1) double {mustBeInteger, mustBeNonnegative}
        mode        (1,1) string {mustBeMember(mode, ["proxy", "test"])}
        xxx
    end

    project_root = fileparts(fileparts(mfilename('fullpath')));
    addpath(genpath(fullfile(project_root, 'matlab')));

    core.validate_spec_structure(spec);

    % Local Spec Setup
    spec_local = spec;
    spec_local.dataset_seed = int32(dataset_seed);

    if mode == "proxy"
        assert(isfield(spec_local, 'unknown_proxy_class_ids'), ...
            'run_pro_test_pipeline: spec.unknown_proxy_class_ids missing.');
        spec_local.class_ids = spec_local.unknown_proxy_class_ids;
        suffix = "proxy";
        spec_local.snr_beta  = 1.9;
        spec_local.snr_alpha = 1.2;
    else
        assert(isfield(spec_local, 'unknown_test_class_ids'), ...
            'run_pro_test_pipeline: spec.unknown_test_class_ids missing.');
        spec_local.class_ids = spec_local.unknown_test_class_ids;
        suffix = "test";
        spec_local.snr_beta  = 1;
        spec_local.snr_alpha = 1;
    end

    version = spec_local.spec_version;

    % SNR override — applies to both modes; tweak as needed
    spec_local.snr_mode = "fixed";
    spec_local.snr_train_db = [-15 15];
    spec_local.snr_eval_db  = [-10 10];
    spec_local.snr_fixed_db = xxx;


    % Impairment generation always runs in eval mode for unknowns,
    % regardless of proxy/test split — proxies still need to look like
    % realistic eval-mode samples to the calibrator.
    impair_mode = 'eval';

    % Folders
    if ~exist('artifacts','dir'); mkdir('artifacts'); end
    output_dir = fullfile('artifacts','datasets','unknown');
    if ~exist(output_dir,'dir'); mkdir(output_dir); end

    fprintf('=== UNKNOWN PIPELINE START (mode=%s) ===\n', suffix);
    fprintf('Seed: %d | n_per_class: %d | classes: [%s] | SNR: %s \n', ...
            dataset_seed, n_per_class, ...
            strjoin(string(spec_local.class_ids), ','), spec_local.snr_mode);

    % Generate Clean Unknowns
    dataset = clean.generate_clean_dataset(n_per_class, spec_local);

    % Apply Impairments
    unknown_data = impaired.generate_impaired_dataset( ...
        dataset, spec_local, impair_mode);

    unknown_data.meta.dataset_version = char("unknown_dataset_v2");
    unknown_data.meta.unknown_role    = char(suffix);

    % Save with mode suffix
    filename = sprintf('unknown_dataset_%s_seed%d_n%d_%s.mat', ...
                        version, dataset_seed, n_per_class, suffix);
    save(fullfile(output_dir, filename), 'unknown_data', '-v7.3');

    % Sanity Checks
    assert(isfield(unknown_data,'meta'), 'Missing meta.');
    assert(unknown_data.meta.dataset_seed == dataset_seed, ...
        'Seed mismatch.');

    fprintf('=== UNKNOWN %s PIPELINE COMPLETE.  ===\n', mode);
end