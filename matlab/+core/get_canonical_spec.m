function spec = get_canonical_spec()
    % GET_CANONICAL_SPEC Returns the parameter spec
 
    % Spec Struct
    %spec.spec_version = "v1";       % MATLAB-PYTHON contract version
    %spec.fs = double(10e6);       % Sampling Frequency (Hz) - 10MHz
    %spec.N = int32(4800);       % Total number of samples 
    %spec.dataset_seed = int32(123);         % For reproducible "randomness"
   
    spec.spec_version = "v2";
    spec.fs = double(10e6);
    spec.N = int32(1024);
    spec.dataset_seed = int32(123); 

end