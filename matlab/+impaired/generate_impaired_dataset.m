function impaired_data = generate_impaired_dataset(clean_dataset, spec, mode)
%GENERATE_IMPAIRED_DATASET Apply v1 impairment to an existing clean dataset.
%
% Inputs
%   clean_dataset : struct with fields:
%                   - X_clean (N x Nsamples)
%                   - y       (Nsamples x 1)
%                   - params  (Nsamples x 1 struct) or struct array
%                   - meta    (struct)
%   spec         : same spec used for clean generation; must include dataset_seed and N
%   mode         : "train" or "eval"
%
% Output
%   impaired : struct with fields:
%              - X_imp
%              - y
%              - params
%              - imp_params
%              - meta
%              - (optional) X_clean if spec.keep_X_clean == true

    arguments
        clean_dataset (1,1) struct
        spec (1,1) struct
        mode (1,:) char
    end

    core.validate_spec_structure(spec);

    Xc = clean_dataset.X_clean;
    y  = clean_dataset.y;

    Ns = size(Xc,2);
    
    % Verify signals are in columns (N x Nsamples)
    N = spec.N;
    assert(size(Xc,1) == N, 'generate_impaired_dataset:BadShape', ...
        'X_clean must be (N x Nsamples) with signals in columns.');
    
    % Transpose to (Nsamples x N) for apply_impairment
    Xc_rows = Xc.'; 

    % Ensure y is column and matches
    y = y(:);
    assert(numel(y) == Ns, 'generate_impaired_dataset:LabelMismatch', ...
        'y length must match number of samples.');
    
    % Pre-allocate
    Ximp = zeros(N, Ns, 'double');
    
    % Initialize imp_params as empty - we'll build it differently
    imp_params(Ns,1) = impaired.init_imp_param_record();
    
    % Apply impairment deterministically to each sample
    for i = 1:Ns
        x_clean_i = Xc_rows(i,:); 
        [x_imp_i, ip] = impaired.apply_impairment(x_clean_i, i-1, spec, mode);
        Ximp(:,i) = x_imp_i;
        imp_params(i) = ip;  % Store in cell array first
    end

    % Build output struct
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
    if isfield(clean_dataset,'meta'), meta = clean_dataset.meta; end
    meta.mode = char(string(lower(mode)));
    meta.dataset_version = char("impaired_dataset_v1");

    % Optional debug retention
    if isfield(spec,'keep_X_clean') && spec.keep_X_clean
        impaired_data.X_clean = Xc_rows;
    end
 
    % Sanity
    assert(all(isfinite(impaired_data.X_imp(:))), 'generate_impaired_dataset:NonFinite', ...
        'X_imp contains NaN/Inf.');

    impaired_data.meta = meta;

     % checksum with hash artifact
    impaired_data.meta.artifact_hash = core.compute_artifact_hash(impaired_data);
    
end