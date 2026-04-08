function impaired_data = generate_impaired_dataset(clean_dataset, spec, mode)

    arguments
        clean_dataset (1,1) struct
        spec (1,1) struct
        mode (1,1) string
    end

    Xc = clean_dataset.X_clean;
    y  = clean_dataset.y;

    Ns = size(Xc,2);


    % Validate shape
    N = spec.N;
    assert(size(Xc,1) == N, ...
        'generate_impaired_dataset:BadShape', ...
        'X_clean must be (N x Nsamples).');

    y = y(:);
    assert(numel(y) == Ns, ...
        'generate_impaired_dataset:LabelMismatch');

    % Preallocate
    Ximp = complex(zeros(N, Ns, 'double'));
    imp_params(Ns,1) = impaired.init_imp_param_record();

    % Deterministic loop
    for i = 1:Ns
        x_clean_i = Xc(:, i);  % column vector
        class_id = int32(y(i));

        [x_imp_i, ip] = impaired.apply_impairment(x_clean_i, class_id, i-1, spec, mode);

        assert(iscolumn(x_imp_i), ...
            'apply_impairment must return column vector.');

        Ximp(:,i) = x_imp_i;
        imp_params(i) = ip;
    end

    % Build struct
    impaired_data = struct();
    impaired_data.X_imp      = Ximp;
    impaired_data.y          = y;

    if isfield(clean_dataset,'params')
        impaired_data.params = clean_dataset.params;
    else
        impaired_data.params = [];
    end

    impaired_data.imp_params = imp_params;

    % Meta
    meta = struct();
    if isfield(clean_dataset,'meta')
        meta = clean_dataset.meta;
    end

    meta.mode = char(mode);
    meta.dataset_version = char("impaired_dataset_v2");
    
    assert(isfield(spec,'snr_mode'), 'Spec must contain snr_mode.');
    meta.snr_mode = spec.snr_mode;

    impaired_data.meta = meta;

    % Sanity
    assert(all(isfinite(impaired_data.X_imp(:))), ...
        'generate_impaired_dataset:NonFinite');

    % Checksum
    impaired_data.meta.artifact_hash = ...
        core.compute_artifact_hash(impaired_data);
end