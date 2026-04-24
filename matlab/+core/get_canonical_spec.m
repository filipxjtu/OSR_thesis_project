function spec = get_canonical_spec()
% GET_CANONICAL_SPEC Returns canonical project specification (v2)

    spec.spec_version = "v2";
    spec.fs = double(10e6);
    spec.N  = int32(1024);
    spec.dataset_seed = int32(123);

    % Class set
    spec.class_ids = int32(0:9);   
    spec.unknown_class_ids = int32(10:13);  

    % SNR control
    spec.snr_mode = "range";           % "range" | "fixed"
    spec.snr_train_db = [-10 5];
    spec.snr_eval_db  = [-5 10];
    spec.snr_fixed_db = -6;
    spec.snr_skew_gamma = 1;

    % Residual oscillator effects
    spec.enable_cfo = true;
    spec.cfo_hz_range = [100, 500];

    spec.enable_phase_noise = true;
    spec.phase_noise_std_range = [0.001, 0.005];

    % Channel model
    spec.enable_channel = true;
    spec.channel_model = "ricean_2tap";

    spec.rice_k_db = 10;
    spec.delay_samp_range = [1, 3];
    spec.echo_gain_db_range = [-12, -6];

end