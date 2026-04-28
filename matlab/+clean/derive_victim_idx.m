function victim_idx = derive_victim_idx(jammer_idx, jammer_class_id)
    % DERIVE_VICTIM_IDX Deterministically generates a unique sample_idx
    % for a coherent target (e.g., LFM) to prevent data leakage.
    
    % Ensure we stay within [0, 999999]
    SAMPLE_IDX_MAX = double(1e6); 
    
    % Prime Spacing
    j_d = double(jammer_idx) * 1000033;
    c_d = double(jammer_class_id) * 1000003;
    unique_idx = j_d + c_d;
    
    % LCG
    mix_d = mod(1664525 * unique_idx + 1013904223, 2^32);
    mix = uint32(mix_d);
    
    % salt
    victim_salt = uint32(429496729); 
    hashed = double(bitxor(mix, victim_salt));
    
    % [0, SAMPLE_IDX_MAX - 1]
    victim_idx = int32(mod(hashed, SAMPLE_IDX_MAX));
    
    % jammer must NEVER use its own index
    if victim_idx == jammer_idx
        % Offset by a large prime
        victim_idx = int32(mod(double(victim_idx) + 123457, SAMPLE_IDX_MAX));
    end
end