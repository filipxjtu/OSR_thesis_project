function validate_active_fields(params, required_fields)
    %VALIDATE_ACTIVE_FIELDS Enforce parameter contract per class.
    % Ensures:
    %   1) All required fields are populated (non-NaN / non-empty)
    %   2) All non-required numeric scalar fields remain NaN
    % required_fields is a string array (may contain nested fields like "bin_info.nfft")

    core_fields = ["class_id","sample_index"];

    % Check required fields are populated
    for i = 1:numel(required_fields)
        field_path = required_fields(i);
        value = get_nested_field(params, field_path);

        assert(~isempty(value), ...
            'Required field "%s" is empty.', field_path);

        if isnumeric(value)
            assert(~(isscalar(value) && isnan(value)), ...
                'Required field "%s" is NaN.', field_path);
        end
    end

    % Check non-required scalar numeric fields remain NaN
    all_fields = fieldnames(params);

    for k = 1:numel(all_fields)
        fname = all_fields{k};

        % skip struct containers
        if isstruct(params.(fname))
            continue;
        end

        % skip if nested field (already handled)
        if any(startsWith(required_fields, fname + "."))
            continue;
        end

        % if field is scalar numeric
        val = params.(fname);
        if isnumeric(val) && isscalar(val)

            if ~any(required_fields == fname) && ~any(core_fields == fname)
                assert(isnan(val), ...
                    'Field "%s" should remain NaN for this class.', fname);
            end
        end
    end
end

% Helper: nested field resolver
function value = get_nested_field(s, field_path)
    parts = split(field_path, ".");
    value = s;

    for i = 1:numel(parts)
        fname = parts{i};
        assert(isfield(value, fname), ...
            'Missing nested field "%s".', field_path);
        value = value.(fname);
    end
end