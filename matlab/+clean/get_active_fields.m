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

        case 3  % SFMJ
            fields = ["A","fc","fm","beta","beta2","phi","phi_m1", "phi_m2"];

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

        case 10 % MBNJ
            fields = ["A", ...
                "mbn_info.frame_len","mbn_info.hop_len","mbn_info.B",...
                "mbn_info.delta_f_sweep","mbn_info.f_start"];

        case 11 % I-OFDMJ
            fields = ["A", ...
                "iofdm_info.D_b_range", "iofdm_info.G_b_range", "iofdm_info.taper_taps"];

        case 12  % DSSS
            fields = ["A", "beta", "Rc", "fc", "phi"];

        case 13  % TFMJ
            fields = ["A", "delta_f", "fm"];

        otherwise
            error("Invalid class_id: %d", class_id);

    end
end