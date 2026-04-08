function validate_active_fields(params, required_fields)
    % VALIDATE_ACTIVE_FIELDS (v2 - Recursive)
    core_fields = ["class_id", "sample_index", "lims", "units"];
    allowed_fields = [required_fields, core_fields];

    % required fields exist and are populated
    for i = 1:numel(required_fields)
        val = get_nested_field(params, required_fields(i));
        assert(~isempty(val) && ~(isnumeric(val) && isscalar(val) && isnan(val)), ...
            'Required field "%s" is missing or NaN.', required_fields(i));
    end

    % everything else must be NaN (Recursive)
    check_unused_is_nan(params, allowed_fields, "");
end

function check_unused_is_nan(node, allowed_paths, current_path)
    % Helper to recursively ensure non-allowed fields remain NaN
    fnames = fieldnames(node);
    
    for i = 1:numel(fnames)
        fname = fnames{i};
        full_path = compose_path(current_path, fname);
        val = node.(fname);

        % check if the specific field is allowed
        is_allowed = any(allowed_paths == full_path);

        if isstruct(val)
            % ignore if it is allowed
            if  ~is_allowed
                check_unused_is_nan(val, allowed_paths, full_path);
            end
            
        elseif ~is_allowed && isnumeric(val) && isscalar(val)
            % It's a leaf node, it's not allowed, it must be NaN
            assert(isnan(val), 'Field "%s" should remain NaN.', full_path);
        end
    end
end

function p = compose_path(parent, child)
    if parent == "", p = string(child); else, p = parent + "." + child; end
end

function value = get_nested_field(s, field_path)
    parts = split(field_path, ".");
    value = s;
    for i = 1:numel(parts)
        assert(isfield(value, parts{i}), 'Missing field "%s".', field_path);
        value = value.(parts{i});
    end
end