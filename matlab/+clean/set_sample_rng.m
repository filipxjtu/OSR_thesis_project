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

    
    base = uint32(mod(double(spec.dataset_seed), 2^32));
    
    class_idx_d = double(class_id) * 1000003;
    sample_idx_d = double(sample_idx) * 1000033;
    unique_idx = (class_idx_d + sample_idx_d);
    
    mix_d = mod(1664525 * unique_idx + 1013904223, 2^32);
    mix = uint32(mix_d);
    
    % Use a unique salt specifically for the clean generator 
    clean_salt = uint32(987654321); 
    
    final_seed = double(bitxor(bitxor(base, mix), clean_salt));

    % Apply the state
    old_state = rng; 
    rng(final_seed, 'twister'); 
end