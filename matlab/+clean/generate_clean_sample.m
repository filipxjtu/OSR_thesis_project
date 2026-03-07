function [x_clean, y, params] = generate_clean_sample(class_id, sample_idx, spec)
    % GENERATE_CLEAN_SAMPLE Orchestrator for single-sample generation.
    
    % Inputs:
    %   class_id   : (int32) The signal category (0-6)
    %   sample_idx : (int32) Unique index for RNG seeding
    %   spec       : Canonical project specification struct
    
    % Outputs:
    %   x_clean    : The synthesized column vector
    %   y          : The label (integer class_id)
    %   params     : The specific parameters used for this waveform

    class_id = int32(class_id);
    sample_idx = int32(sample_idx);

    % generate parameters
    params = clean.generate_sample_params(class_id, sample_idx, spec);

    switch class_id
        case 0
            x_clean = clean.synthesize_clean_signal_class0(params, spec);
        case 1
            x_clean = clean.synthesize_clean_signal_class1(params, spec);
        case 2
            x_clean = clean.synthesize_clean_signal_class2(params, spec);
        case 3
            x_clean = clean.synthesize_clean_signal_class3(params, spec);
        case 4
            x_clean = clean.synthesize_clean_signal_class4(params, spec);
        case 5
            x_clean = clean.synthesize_clean_signal_class5(params, spec);
        case 6
            x_clean = clean.synthesize_clean_signal_class6(params, spec);
        otherwise
            error('Invalid class_id: %d. Must be 0-6.', class_id);
    end

    x_clean = x_clean - mean(x_clean);

    % set label
    y = int32(class_id);

    % boundary assertions
    assert(numel(x_clean) == spec.N, 'Signal length must match spec.N');
    assert(iscolumn(x_clean), 'Signal must be a column vector');
    assert(params.class_id == y, 'Parameter class mismatch with label y');
end