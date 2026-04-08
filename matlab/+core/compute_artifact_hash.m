function hash = compute_artifact_hash(dataset)
% Simple deterministic 64-bit checksum over X and y

    if isfield(dataset,'X_clean')
        X = dataset.X_clean;
    elseif isfield(dataset,'X_imp')
        X = dataset.X_imp;
    else
        error('compute_artifact_hash:NoSignalField', ...
              'Dataset must contain X_clean or X_imp.');
    end

    y = dataset.y;

    % Concatenate numeric content deterministically
    if ~isreal(X)
        data = [real(X(:)); imag(X(:)); double(y(:))];
    else
        data = [X(:); double(y(:))];
    end

    % Convert to bytes
    bytes = typecast(data,'uint8');

    % 64-bit accumulation with wrap
    hash = uint64(14695981039346656037);

    for i = 1:numel(bytes)
        hash = hash + uint64(bytes(i));
    end

    % Force wrap to 64-bit
    %hash = uint64(mod(double(hash), 2^64));
end