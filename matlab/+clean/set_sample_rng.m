function [old_state] = set_sample_rng(spec, class_id, sample_idx)
    % SET_SAMPLE_RNG Sets a deterministic RNG state for a specific sample.
    % Returns the current state so you can restore it later.
    
      % Seed layout:
    % [dataset_seed][class_id][sample_idx]
    % dataset_seed : up to 3 digits
    % class_id     : up to 3 digits
    % sample_idx   : up to 6 digits

    
    %asserting
    assert(isa(spec.dataset_seed, 'int32'), 'spec.dataset_seed must be int32');
    assert(isa(class_id, 'int32'), 'class_id must be int32');
    assert(isa(sample_idx, 'int32'), 'sample_idx must be int32');
    assert(sample_idx < int32(1e6), 'sample_idx exceeds seed allocation');

    % Seed (0-999), ClassID (0-999), SampleIdx (0-999,999) for uique seed
    combined_seed = spec.dataset_seed * 1e7 + ...
                    class_id * 1e5 + ...
                    sample_idx;
                
    % rng expects 32-bit unsigned int
    final_seed = mod(uint32(combined_seed), 2^32);

    % Apply the state
    old_state = rng; 
    rng(final_seed, 'twister'); 
end