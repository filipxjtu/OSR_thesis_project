function validate_spec_structure(spec)
%VALIDATE_SPEC_STRUCTURE Ensure canonical spec integrity.

    assert(isstruct(spec), 'Spec must be a struct.');

    required = ["spec_version","fs","N","dataset_seed"];

    for i = 1:numel(required)
        assert(isfield(spec, required(i)), ...
            'Spec missing required field "%s".', required(i));
    end

    assert(isstring(spec.spec_version) || ischar(spec.spec_version), ...
        'spec_version must be string.');

    assert(isnumeric(spec.fs) && isscalar(spec.fs) && spec.fs > 0, ...
        'fs must be positive scalar.');

    assert(isnumeric(spec.N) && isscalar(spec.N) && spec.N > 0, ...
        'N must be positive scalar.');

    assert(isa(spec.dataset_seed,'int32'), ...
        'dataset_seed must be int32.');
end