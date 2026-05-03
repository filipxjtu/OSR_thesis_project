function fields = get_active_fields(class_id)
    % GET_ACTIVE_FIELDS
    % Extracted directly from generate_sample_params (ground-truth aligned)

    class_id = int32(class_id);

    switch class_id

        case 0  % STJ
            fields = ["A","f0","phi","delta_f","sigma"];

        case 1  % MTJ
            fields = ["A","f0","phi","K","alpha","tau_ns"];

        case 2  % LFMJ
            fields = ["A", "fc", "f0","f1","phi","T","K","delta","alpha_tukey"];

        case 3  % MBNJ
            fields = ["A", ...
                "mbn_info.frame_len","mbn_info.hop_len","mbn_info.B",...
                "mbn_info.delta_f_sweep","mbn_info.f_start"];

        case 4  % PBNJ
            fields = ["A","fc","fL","fH","bandwidth","alpha","delta","theta", ...
                "filter_info.filter_order","filter_info.Vsat_factor"];

        case 5  % FHJ
            fields = ["A", ...
                "hop_info.f_grid", "hop_info.hop_idx", ...
                "hop_info.phi_h", "hop_info.Lh", ...
                "hop_info.H", "hop_info.P_trans", "hop_info.N_trans"];

        case 6 % OFDMJ
            fields = ["A","fL","fH","fc","phi", ...
                "ofdm_info.Nfft", "ofdm_info.Lcp","ofdm_info.Nsym",...
                "ofdm_info.active_ratio","ofdm_info.Nc","ofdm_info.M",...
                "ofdm_info.Amax_factor","ofdm_info.delta_f","ofdm_info.B_occ"];

        case 7 % PGPJ
            fields = ["A", "fc", "phi", "alpha", "sigma", "beta", ...
                "pulse_info.eta", "pulse_info.T_rep", "pulse_info.M", ...
                "pulse_info.c_ideal", "pulse_info.epsilon", "pulse_info.centers"];

        case 8  % ISRJ
            fields = ["A","K", ...
                "srj_info.D", "srj_info.gap", "srj_info.T_pri", ...
                "srj_info.M", "srj_info.q", "srj_info.L", ...
                "srj_info.gamma", "srj_info.epsilon",... 
                "srj_info.use_additive_overlap", "srj_info.target_type"];

        case 9  % DFTJ
            fields = ["A", ...
                "dftj_info.Q", "dftj_info.q", "dftj_info.L",...
                "dftj_info.tau", "dftj_info.delta_f", "dftj_info.A_q"];

        case 10  % IASNJ
            fields = ["A", "alpha", "beta", "gamma", "delta"];

        case 11 % CCJ
            fields = ["A", "fc", "phi", "Rc", "sigma", "alpha"];

        case 12  % DSSSJ
            fields = ["A", "beta", "Rc", "fc", "phi"];

        case 13  % PCPJ
            fields = ["A", "fc", "phi",...
                "pulse_info.code_type", "pulse_info.T_pulse", ...
                "pulse_info.M", "pulse_info.N_chips",...
                "pulse_info.code_phase", "pulse_info.t_start", "pulse_info.PRF"];

        case 14  % TFMJ
            fields = ["A", "delta_f", "fm", "K"];

        case 15  % RBJ
            fields = ["A", "fc",...
                "burst_info.B", "burst_info.T_on_mean", "burst_info.T_on_std",...
                "burst_info.mu_ln", "burst_info.sigma_ln", "burst_info.M_max", ];

        case 16  % PGNJ
            fields = ["A", ...
                "pgn_info.PRF", "pgn_info.duty_cycle", "pgn_info.rise_samp"];

        case 17 % SFMJ
            fields = ["A","fc","fm","beta","beta2","phi","phi_m1", "phi_m2"];


        otherwise
            error("Invalid class_id: %d", class_id);

    end
end