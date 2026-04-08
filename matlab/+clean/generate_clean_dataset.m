function dataset = generate_clean_dataset(n_per_class, spec)
    
    class_set = spec.class_ids;
    num_classes = numel(class_set);
    TotalSamples = num_classes * n_per_class;

    % Preallocate correctly
    X_clean = complex(zeros(spec.N, TotalSamples));
    y = zeros(TotalSamples, 1, 'int32');
    params(TotalSamples,1) = clean.init_clean_param_record(0, 0);

    % Deterministic generation loop
    idx = 1;
    for class_id = class_set
        for sample_idx = 0:(n_per_class-1)

            [x_clean, label, p] = clean.generate_clean_sample(class_id, sample_idx, spec);

            % Enforce column orientation
            if isrow(x_clean)
                x_clean = x_clean.';
            end

            X_clean(:, idx) = double(x_clean);
            y(idx, 1)       = int32(label);
            params(idx) = p;

            idx = idx + 1;
        end
    end
    
    % Assemble dataset struct
    dataset = struct();
    dataset.X_clean = X_clean;
    dataset.y       = y;
    dataset.params  = params;

    dataset.meta = struct();
    dataset.meta.spec_version      = char(spec.spec_version);
    dataset.meta.dataset_seed      = spec.dataset_seed;
    dataset.meta.artifact_hash_fn  = char("simple64_checksum");
    dataset.meta.layout            = char("N_by_Ns_columns_are_samples");
    dataset.meta.dtype_policy      = char("complex128_X_int32_y");
    dataset.meta.N                 = spec.N;
    dataset.meta.fs                = spec.fs;
    dataset.meta.Ns                = TotalSamples;
    dataset.meta.n_per_class       = n_per_class;
    dataset.meta.class_set         = class_set;
    dataset.meta.dataset_version   = char("clean_dataset_v2");
    dataset.meta.created_utc       = char(string(datetime("now","TimeZone","UTC","Format","yyyy-MM-dd'T'HH:mm:ss'Z'")));
    
    % checksum with hash artifact
    dataset.meta.artifact_hash = core.compute_artifact_hash(dataset);
    
    % Validation block 
    assert(all(isfinite(dataset.X_clean(:))), 'Dataset contains non-finite values.');
    assert(size(dataset.X_clean,1) == spec.N, 'Incorrect signal length.');
    assert(numel(dataset.y) == size(dataset.X_clean,2), 'Label mismatch.');
end