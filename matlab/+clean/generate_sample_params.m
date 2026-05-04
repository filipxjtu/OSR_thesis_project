function params = generate_sample_params(class_id, sample_idx, spec)
    % GENERATE_SAMPLE_PARAMS Samples reproducible signal parameters
    % return parameters are determined based on the class_id

    %enforcing accepted type
    class_id   = int32(class_id);
    sample_idx = int32(sample_idx);

    % set_sample_rng assertions.
    old_state = clean.set_sample_rng(spec, class_id, sample_idx);

    % common metadata
    params = clean.init_clean_param_record(class_id,sample_idx);
    
  % class_id based sampling
    switch class_id
        case 0  % Single Tone (Clean)
            % Constraints
            lims.A        = [0.8, 1.2];
            lims.f0       = [0.5e6, 4.5e6];
            lims.phi      = [0, 2*pi];
            lims.delta_f  = [0, 50e3];         % magnitude range
            lims.sigma    = [0.01, 0.05];      % phase noise diffusion (rad/√Hz)
            lims.p_zero   = 0.2;            % 20% possibility fo discrete spike (co-channel)
 
            % Sample amplitude
            params.A = lims.A(1) + diff(lims.A) * rand(1);
            params.phi = lims.phi(1) + diff(lims.phi) * rand(1);
            
            % Signed frequency offset (constant offset)
            if rand < lims.p_zero  
                params.delta_f = lims.delta_f(1);
            else
                params.delta_f =(2*(rand > 0.5)-1) * (lims.delta_f(2) * rand);
            end

            % Nominal frequency (ensure total stays within band)
            f0_min_valid = max(lims.f0(1), lims.f0(1) - params.delta_f);
            f0_max_valid = min(lims.f0(2), lims.f0(2) - params.delta_f);

            params.f0 = f0_min_valid + (f0_max_valid - f0_min_valid) * rand(1);
      
            % Wiener phase noise step std
            params.sigma = lims.sigma(1) + diff(lims.sigma) * rand(1);
            
            params.lims = lims;
            
            % Units
            params.units.A        = char("linear");
            params.units.f0       = char("Hz");
            params.units.phi      = char("rad");
            params.units.delta_f  = char("Hz");
            params.units.sigma    = char("rad/√Hz");
            
        case 1  % Multi-Tone (Clean)
            % Constraints
            lims.K_range        = [4, 6];
            lims.A_total        = [0.8, 1.2];
            lims.w              = [0.6, 1.0];
            lims.f_band         = [0.6e6, 3.2e6];
            lims.delta_f_min    = 0.20e6;
            lims.delta_f_max    = 0.60e6;
            lims.total_span_max = 1.80e6;
            lims.phi            = [0, 2*pi];
            lims.alpha          = [0.02, 0.08];
            lims.tau_ns         = [50, 300];   % weak to moderate memory effects

            % Number of tones
            params.K = randi(lims.K_range);

            % Total amplitude
            A_total = lims.A_total(1) + diff(lims.A_total) * rand(1);

            % Generate valid tone frequencies
            valid = false;
            while ~valid
                f_anchor = 0.80e6 + (2.80e6 - 0.80e6) * rand(1);

                offsets = zeros(params.K, 1);
                for k = 2:params.K
                    df = lims.delta_f_min + ...
                         (lims.delta_f_max - lims.delta_f_min) * rand(1);
                    offsets(k) = offsets(k-1) + df;
                end

                % center around anchor
                offsets = offsets - mean(offsets);
                f_vec = f_anchor + offsets;

                valid = min(f_vec) >= lims.f_band(1) && ...
                        max(f_vec) <= lims.f_band(2) && ...
                        (max(f_vec) - min(f_vec)) <= lims.total_span_max;
            end

            params.f0 = sort(f_vec);

            % Tone amplitudes & phases
            w = lims.w(1) + diff(lims.w) * rand(params.K, 1);
            params.A = A_total * w / sqrt(sum(w.^2));
            params.phi = lims.phi(1) + diff(lims.phi) * rand(params.K, 1);

            % PA cubic coefficient & memory parameter
            params.alpha = lims.alpha(1) + diff(lims.alpha) * rand(1);
            params.tau_ns = lims.tau_ns(1) + diff(lims.tau_ns) * rand(1);

            params.lims = lims;

            % Units
            params.units.A       = char("linear vector");
            params.units.f0      = char("Hz vector");
            params.units.phi     = char("rad vector");
            params.units.K       = char("integer");
            params.units.alpha   = char("ratio");
            params.units.tau_ns  = char("ns");

        case 2  % LFM sweep (Clean)
            % Constraints
            lims.A   = [0.8, 1.2];
            lims.fc  = [0.5e6, 4.5e6];
            lims.B   = [1e6, 3.5e6];
            lims.T_obs = [1e-4, 1.5e-4]; % allows partial observation
            lims.phi = [0, 2*pi];

            lims.alpha_tukey = [0.1, 0.25];
            lims.delta_mag   = [1e-5, 5e-5];   % magnitude only (sign later)

            % Sample basic params
            params.A   = lims.A(1) + diff(lims.A) * rand(1);
            params.phi = lims.phi(1) + diff(lims.phi) * rand(1);

            % Sample bandwidth first
            B = lims.B(1) + diff(lims.B) * rand(1);

            % Sample fc with validity
            fc_min_valid = lims.fc(1) + B/2;
            fc_max_valid = lims.fc(2) - B/2;

            params.fc = fc_min_valid + (fc_max_valid - fc_min_valid) * rand(1);

            % Assign sweep endpoints
            params.f0 = params.fc - B/2;
            params.f1 = params.fc + B/2;

            % Observation duration
            params.T = lims.T_obs(1) + diff(lims.T_obs) * rand(1);

            % Tukey window
            params.alpha_tukey = lims.alpha_tukey(1) + ...
                                 diff(lims.alpha_tukey) * rand(1);

            % Doppler scaling (magnitude + random sign)
            delta_mag = lims.delta_mag(1) + diff(lims.delta_mag) * rand(1);
            delta_sign = 2 * (rand > 0.5) - 1;
            params.delta = delta_sign * delta_mag;

            % Derived chirp rate
            params.K = (params.f1 - params.f0) / params.T;

            % Structural assertions
            assert(params.T > 0, 'LFM: T must be positive.');
            assert(abs(params.f1 - params.f0) >= 1e6, ...
                'LFM: bandwidth constraint violated.');

            params.lims = lims;

            % Units
            params.units.A   = char("linear");
            params.units.f0  = char("Hz");
            params.units.f1  = char("Hz");
            params.units.T   = char("seconds");
            params.units.K   = char("Hz/s");
            params.units.phi = char("rad");
            params.units.alpha_tukey = char("ratio");
            params.units.delta = char("ratio");
        
        case 3 % SFM sweep (Clean)
            % Constraints
            lims.A    = [0.8, 1.2];
            lims.fc   = [0.5e6, 4.5e6];
            lims.fm   = [20e3, 80e3];
            lims.beta = [5, 15];
            lims.beta2_ratio = [0.1, 0.3];
            lims.phi_range  = [0, 2*pi];

            % Sample
            params.A   = lims.A(1) + diff(lims.A) * rand(1);
            params.fm  = lims.fm(1) + diff(lims.fm) * rand(1);
            params.beta = lims.beta(1) + diff(lims.beta) * rand(1);

            beta2_ratio = lims.beta2_ratio(1) + ...
                          diff(lims.beta2_ratio) * rand(1);

            params.beta2 = beta2_ratio * params.beta;

            % Carrier frequency
            max_dev = params.fm * (params.beta + 2 * params.beta2);

            fc_min_valid = lims.fc(1) + max_dev;
            fc_max_valid = lims.fc(2) - max_dev;

            params.fc = fc_min_valid + (fc_max_valid - fc_min_valid) * rand(1);

            % sampling the phases
            params.phi   = lims.phi_range(1) + diff(lims.phi_range) * rand(1);
            params.phi_m1 = lims.phi_range(1) + diff(lims.phi_range) * rand(1);
            params.phi_m2 = lims.phi_range(1) + diff(lims.phi_range) * rand(1);

            % Structural assertions
            assert(params.fm > 0, 'SFM: fm must be positive.');
            assert(params.beta2 > 0, 'SFM: beta2 must be positive.');

            params.lims = lims;

            % Units
            params.units.A      = char("linear");
            params.units.fc     = char("Hz");
            params.units.fm     = char("Hz");
            params.units.beta  = char("ratio");
            params.units.beta2  = char("ratio");
            params.units.phi    = char("rad");
            params.units.phi_m1 = char("rad");
            params.units.phi_m2 = char("rad");

        case 4  % Partial Band Noise Jamming (Clean)
            % Constraints
            lims.A   = [0.8, 1.2];
            lims.fc  = [0.5e6, 4.5e6];
            lims.B   = [1.5e6, 2.5e6];
            lims.alpha = [0.89, 1.12]; % IQ gain imbalance (±1 dB)
            lims.theta = [0.035, 0.175];  % IQ phase imbalance (2° to 10° rad)
            lims.delta = [0.01, 0.0316]; % Carrier leakage amplitude (-40 to -30 dBc)

            params.A = lims.A(1) + diff(lims.A) * rand(1);
            B = lims.B(1) + diff(lims.B) * rand(1);

            % Sample fc with validity
            fc_min_valid = lims.fc(1) + B/2;
            fc_max_valid = lims.fc(2) - B/2;

            params.fc = fc_min_valid + (fc_max_valid - fc_min_valid) * rand(1);

            % Assign sweep endpoints
            params.fL = params.fc - B/2;
            params.fH = params.fc + B/2;
            params.bandwidth = B;


            % Sample hardware imperfection parameters
            params.alpha = lims.alpha(1) + diff(lims.alpha) * rand(1);
            params.theta = lims.theta(1) + diff(lims.theta) * rand(1);
            params.delta = lims.delta(1) + diff(lims.delta) * rand(1);

            % Fixed design parameters
            filter_info.filter_order = 4;
            filter_info.Vsat_factor = 2.0;

            params.lims = lims;
            params.filter_info = filter_info;

            % Units
            params.units.A          = char("linear");
            params.units.fc         = char("Hz");
            params.units.bandwidth  = char("Hz");
            params.units.fL         = char("Hz");
            params.units.fH         = char("Hz");
            params.units.alpha      = char("ratio");
            params.units.theta      = char("radians");
            params.units.delta      = char("linear ratio");
            params.units.filter_order = char("integer");
            params.units.Vsat_factor  = char("ratio");

        case 5  % Frequency Hopping Jamming (Clean)
            % Constraints
            lims.A = [0.8, 1.2];
            lims.f_range = [0.5e6, 4.5e6];
            lims.delta_f_grid = 0.2e6;

            lims.H = [6, 9];
            lims.P_trans = [0.02, 0.05];
            lims.phi_range = [0, 2*pi];

            % Sample amplitude
            params.A = lims.A(1) + diff(lims.A) * rand(1);

            % Frequency grid (discrete)
            f_grid = (lims.f_range(1):lims.delta_f_grid:lims.f_range(2))';
            hop_info.f_grid = f_grid;

            % Number of hops
            H = randi(lims.H);
            hop_info.H = H;

            % Dwell length (derived)
            N = double(spec.N);
            Lh = floor(N / H) * ones(H,1);

            % Fix remainder
            Lh(end) = Lh(end) + (N - sum(Lh));
            assert(sum(Lh) == N, 'FHJ: dwell mismatch.');

            hop_info.Lh = Lh;

            % Sample hop indices (no dedimmediate repetition)
            hop_idx = zeros(H,1);
            for i = 1:H
                valid = false;
                while ~valid
                    idx = randi(numel(f_grid));
                    if i == 1 || idx ~= hop_idx(i-1)
                        hop_idx(i) = idx;
                        valid = true;
                    end
                end
            end

            hop_info.hop_idx = hop_idx;

            % Phase per hop
            hop_info.phi_h = lims.phi_range(1) + ...
                           diff(lims.phi_range) * rand(H,1);

            % Transient proportions
            hop_info.P_trans = lims.P_trans(1) + ...
                             diff(lims.P_trans) * rand(H,1);

            % Derived switching samples
            hop_info.N_trans = round(hop_info.P_trans .* Lh);

            params.lims = lims;
            params.hop_info = hop_info;

            % Units
            params.units.A        = char("linear");
            params.units.f_grid   = char("Hz vector");
            params.units.hop_idx  = char("index vector");
            params.units.Lh       = char("samples vector");
            params.units.H        = char("integer");
            params.units.phi_h    = char("rad vector");
            params.units.P_trans  = char("ratio vector");
            params.units.N_trans  = char("samples vector");
        
        case 6  % OFDM Jamming (Clean)
            % Constraints
            lims.A            = [0.8, 1.2];
            lims.Nfft_choices = [128, 256];
            lims.Lcp_choices  = [1/8, 1/4];
            lims.active_ratio_range = [0.6, 0.9];
            lims.fc_range      = [0.1, 0.9];  % ratio of max_allowed_fc
            lims.phi_range    = [0, 2*pi];
            lims.Amax_factor  = [1.4, 1.8];
            
            % Sample amplitude
            params.A = lims.A(1) + diff(lims.A) * rand(1);
            
            % Draw OFDM geometry
            ofdm_info.Nfft = lims.Nfft_choices(randi(numel(lims.Nfft_choices)));
            
            Lcp_ratio = lims.Lcp_choices(randi(numel(lims.Lcp_choices)));
            ofdm_info.Lcp = round(Lcp_ratio * ofdm_info.Nfft);
            ofdm_info.Nsym = ofdm_info.Nfft + ofdm_info.Lcp;
            
            % Active subcarriers (random percentage of Nfft)
            active_ratio = lims.active_ratio_range(1) + ...
                           diff(lims.active_ratio_range) * rand(1);
            ofdm_info.Nc = round(active_ratio * ofdm_info.Nfft);
            ofdm_info.active_ratio = active_ratio;
            
            % Make Nc even so it splits equally around DC
            if mod(ofdm_info.Nc, 2) ~= 0
                ofdm_info.Nc = ofdm_info.Nc - 1;
            end
            
            % Ensure at least 2 active subcarriers
            if ofdm_info.Nc < 2
                ofdm_info.Nc = 2;
            end
            
            assert(ofdm_info.Nc <= ofdm_info.Nfft - 2, ...
                   'OFDM: Nc must be <= Nfft-2 (need DC and at least one guard bin).');
            
            % Number of OFDM symbols needed
            ofdm_info.M = ceil(double(spec.N) / double(ofdm_info.Nsym));
            
            % Soft clipping factor
            ofdm_info.Amax_factor = lims.Amax_factor(1) + ...
                                    diff(lims.Amax_factor) * rand(1);
            
            % Carrier frequency and occupied bandwidth
            delta_f = double(spec.fs) / double(ofdm_info.Nfft);
            ofdm_info.delta_f = delta_f;
            ofdm_info.B_occ = ofdm_info.Nc * delta_f;

            % get maximum allowed fc to prevent aliasing
            max_allowed_fc = (spec.fs/2) - (ofdm_info.B_occ/2);
    
            % Carrier frequency 
            fc_ratio = lims.fc_range(1) + diff(lims.fc_range) * rand(1);
            params.fc = fc_ratio * max_allowed_fc; 
           
            % Calculate actual spectrum edges (for info only)
            params.fL = params.fc - ofdm_info.B_occ/2;
            params.fH = params.fc + ofdm_info.B_occ/2;
            
            % Initial phase
            params.phi = lims.phi_range(1) + ...
                         diff(lims.phi_range) * rand(1);
            
            params.lims = lims;
            params.ofdm_info = ofdm_info;
            
            % Units
            params.units.A           = char("linear");
            params.units.Nfft        = char("samples");
            params.units.Lcp         = char("samples");
            params.units.Nsym        = char("samples");
            params.units.Nc          = char("integer");
            params.units.M           = char("integer");
            params.units.Amax_factor = char("ratio");
            params.units.delta_f     = char("Hz");
            params.units.B_occ       = char("Hz");
            params.units.fc          = char("Hz");
            params.units.fL          = char("Hz");
            params.units.fH          = char("Hz");
            params.units.phi         = char("rad");
            
        case 7  % Periodic Gaussian Pulse Jamming (Clean)
            % Constraints
            lims.A         = [0.8, 1.2];
            lims.T_rep     = [100, 250];      % samples
            lims.alpha     = [0.07, 0.15];    % sigma = alpha * T_rep
            lims.beta      = [0.05, 0.20];    % jitter ratio
            lims.eta       = [0.85, 1.15];    % per-pulse fading
            lims.fc        = [1.0e6, 3.5e6];
            lims.phi       = [0, 2*pi];

            % Amplitude
            params.A = lims.A(1) + diff(lims.A) * rand(1);

            % Repetition period
            pulse_info.T_rep = randi(lims.T_rep);

            % Number of pulses in observation
            pulse_info.M = floor(double(spec.N) / double(pulse_info.T_rep));
            assert(pulse_info.M >= 4, 'PGPJ: M must be at least 4.');

            % Width control
            params.alpha = lims.alpha(1) + diff(lims.alpha) * rand(1);
            params.sigma = params.alpha * pulse_info.T_rep;

            % Jitter control
            params.beta = lims.beta(1) + diff(lims.beta) * rand(1);

            % Carrier and phase
            params.fc  = lims.fc(1) + diff(lims.fc) * rand(1);
            params.phi = lims.phi(1) + diff(lims.phi) * rand(1);

            % Per-pulse amplitude variation
            pulse_info.eta = lims.eta(1) + diff(lims.eta) * rand(pulse_info.M, 1);

            % Ideal centers and jittered centers
            pulse_info.c_ideal = zeros(pulse_info.M, 1);
            pulse_info.epsilon = zeros(pulse_info.M, 1);
            pulse_info.centers = zeros(pulse_info.M, 1);

            for m = 1:pulse_info.M
                c_ideal = m * double(pulse_info.T_rep);
                epsilon = (-params.beta + 2 * params.beta * rand(1)) * double(pulse_info.T_rep);
                center = c_ideal + epsilon;

                pulse_info.c_ideal(m) = c_ideal;
                pulse_info.epsilon(m) = epsilon;
                pulse_info.centers(m) = center;
            end

            params.pulse_info = pulse_info;
            params.lims = lims;

            % Units
            params.units.A        = char("linear");
            params.units.T_rep    = char("samples");
            params.units.M        = char("integer");
            params.units.alpha    = char("ratio");
            params.units.sigma    = char("samples");
            params.units.beta     = char("ratio");
            params.units.fc       = char("Hz");
            params.units.phi      = char("rad");
            params.units.eta      = char("linear vector");
            params.units.c_ideal  = char("sample index vector");
            params.units.epsilon  = char("samples vector");
            params.units.centers  = char("samples vector");

        case 8  % Sliced-Repeating Jamming / ISRJ (Clean)
            % Constraints
            lims.A            = [0.8, 1.2];
            lims.D            = [40, 70];          % slice duration (samples)
            lims.K_choices    = [3, 4];         % repeats per slice
            lims.gap          = [10, 40];          % processing latency (samples)
            lims.q_choices    = 4;            % ADC bit depth
            lims.gamma        = [0.9, 1.1];        % pulse-to-pulse amplitude drift
            lims.epsilon_vals = [-1, 0, 1];        % discrete timing jitter (samples)

            % Global amplitude
            params.A = lims.A(1) + diff(lims.A) * rand(1);

            % Target selection
            srj_info.target_type = char("lfm");
            %if rand(1) < 0.5
                %srj_info.target_type = char("lfm");
            %else
                %srj_info.target_type = char("ofdm");
            %end

            % DRFM geometry
            params.K = lims.K_choices(randi(numel(lims.K_choices)));
            srj_info.D = randi(lims.D);
            srj_info.gap = randi(lims.gap);

            srj_info.T_pri = srj_info.D * (params.K + 1) + srj_info.gap;
            srj_info.M = floor(double(spec.N) / double(srj_info.T_pri));
            assert(srj_info.M >= 1, 'ISRJ: M must be at least 1.');

            % Quantization
            srj_info.q = lims.q_choices(randi(numel(lims.q_choices)));
            srj_info.L = 2^(srj_info.q - 1) - 1;   % signed symmetric quantizer level index
            assert(srj_info.L >= 1, 'ISRJ: invalid quantizer levels.');

            % Per-period amplitude drift
            srj_info.gamma = lims.gamma(1) + diff(lims.gamma) * rand(srj_info.M, 1);

            % Per-repeat timing jitter
            srj_info.epsilon = zeros(srj_info.M, params.K);
            for m = 1:srj_info.M
                for k = 1:params.K
                    srj_info.epsilon(m, k) = lims.epsilon_vals(randi(numel(lims.epsilon_vals)));
                end
            end

            srj_info.use_additive_overlap = true;

            params.srj_info = srj_info;
            params.lims = lims;

            % Units
            params.units.A         = char("linear");
            params.units.D         = char("samples");
            params.units.K         = char("integer");
            params.units.gap       = char("samples");
            params.units.T_pri     = char("samples");
            params.units.M         = char("integer");
            params.units.q         = char("bits");
            params.units.L         = char("integer");
            params.units.gamma     = char("linear vector");
            params.units.epsilon   = char("samples matrix");
            params.units.target_type = char("string");

        case 9  % Digital False Target Jamming (DFTJ)
            % Constraints
            lims.A          = [0.9, 1.1];
            lims.Q_choices  = [4, 5];
            lims.tau        = [10, 400];          % delay (samples)
            lims.delta_f    = [80e3, 250e3];      % Doppler (Hz)
            lims.q_choices  = [3, 4];             % ADC bits
            lims.tau_min    = 20;                 % minimum separation

            % amplitude
            params.A = lims.A(1) + diff(lims.A) * rand(1);

            % number of targets
            dftj_info.Q = lims.Q_choices(randi(numel(lims.Q_choices)));

            % quantization
            dftj_info.q = lims.q_choices(randi(numel(lims.q_choices)));
            dftj_info.L = 2^(dftj_info.q - 1) - 1;

            % delays with minimum separation
            tau_vals = zeros(dftj_info.Q,1);
            count = 0;
            while count < dftj_info.Q
                candidate = randi(lims.tau);
                if count == 0 || all(abs(candidate - tau_vals(1:count)) >= lims.tau_min)
                    count = count + 1;
                    tau_vals(count) = candidate;
                end
            end

            % doppler magnitudes
            mag = lims.delta_f(1) + diff(lims.delta_f) * rand(dftj_info.Q,1);

            % doppler signs (force diversity)
            signs = ones(dftj_info.Q,1);
            if dftj_info.Q >= 2
                signs(1) = 1;
                signs(2) = -1;
                for i = 3:dftj_info.Q
                    signs(i) = sign(randn);
                    if signs(i) == 0
                        signs(i) = 1;
                    end
                end
            end
            
            % final doppler values
            delta_f_vals = mag .* signs;

            % --- Rayleigh amplitudes ---
            sigma = 1.0;
            A_q = sigma * sqrt(-2 * log(1 - rand(dftj_info.Q,1)));

            % Store
            dftj_info.tau      = tau_vals;
            dftj_info.delta_f  = delta_f_vals;
            dftj_info.A_q      = A_q;

            params.dftj_info = dftj_info;
            params.lims = lims;

            % Units
            params.units.A        = char("linear");
            params.units.Q        = char("integer");
            params.units.tau      = char("samples");
            params.units.delta_f  = char("Hz");
            params.units.q        = char("bits");
            params.units.L        = char("integer");
            params.units.A_q      = char("linear vector");

        case 10  % Impulsive α‑Stable Noise Jammer  unknown 1
            % Constraints
            lims.A          = [0.8, 1.2];
            lims.alpha      = [1.1, 1.4];     % impulsive regime 
            lims.beta       = [-0.2, 02];     % Slight skewness/asymmetry
            lims.gamma      = [0.5, 2.0];     % Scale parameter
            lims.delta      = [-0.1, 0.1];    % Minor location/DC offset

            params.A     = lims.A(1) + diff(lims.A) * rand;
            params.alpha = lims.alpha(1) + diff(lims.alpha) * rand;
            params.gamma = lims.gamma(1) + diff(lims.gamma) * rand;
            params.beta  = lims.beta(1) + diff(lims.beta) * rand;
            params.delta = lims.delta(1) + diff(lims.delta) * rand;

            params.lims = lims;
            params.units.A     = char("linear");
            params.units.alpha = char("ratio (1,2]");
            params.units.beta  = char("ratio");
            params.units.gamma = char("scale");
            params.units.delta = char("location");          

          case 11  % Chaotic FM Jammer (logistic map drives instantaneous frequency)
            % Constraints
            lims.A          = [0.8, 1.2];
            lims.fc         = [0.5e6, 4.5e6];
            lims.phi_range  = [0, 2*pi];
            lims.r_logistic = [3.92, 3.99];   % chaotic regime (avoid period windows)
            lims.f_dev      = [150e3, 500e3]; % chaotic frequency excursion (Hz)
            lims.y0         = [0.15, 0.85];   % initial condition (avoid 0, 0.5, 1)

            % Sample core parameters
            params.A   = lims.A(1) + diff(lims.A) * rand(1);
            params.phi = lims.phi_range(1) + diff(lims.phi_range) * rand(1);
            params.Rc  = lims.r_logistic(1) + diff(lims.r_logistic) * rand(1);   % logistic r
            params.sigma = lims.f_dev(1) + diff(lims.f_dev) * rand(1);           % freq deviation (Hz)
            params.alpha = lims.y0(1) + diff(lims.y0) * rand(1);                 % logistic y0

            % fc must leave room for the chaotic excursion on both sides
            fc_min_valid = lims.fc(1) + params.sigma;
            fc_max_valid = lims.fc(2) - params.sigma;
            params.fc = fc_min_valid + (fc_max_valid - fc_min_valid) * rand(1);

            params.lims = lims;

            % Units (note semantic shift from previous version)
            params.units.A     = char("linear");
            params.units.fc    = char("Hz");
            params.units.phi   = char("rad");
            params.units.Rc    = char("logistic-r (chaos param)");
            params.units.sigma = char("Hz (chaotic freq deviation)");
            params.units.alpha = char("ratio (logistic y0)");
            
        case 12  % Reactive Burst Jammer (narrowband, bursty) Unknown Class 3
            % Constraints
            lims.A               = [0.8, 1.2];
            lims.fc              = [0.5e6, 4.5e6];
            lims.B               = [100e3, 500e3];      % narrowband burst
            lims.T_on_mean       = [16, 48];             % samples
            lims.T_on_std        = [3, 8];              % samples
            lims.mu_ln           = [log(20e-6), log(80e-6)];   % log of inter-arrival mean (seconds)
            lims.sigma_ln        = [0.4, 0.9];          % shape parameter
            lims.M_max           = [8, 25];

            params.A     = lims.A(1) + diff(lims.A) * rand;
            params.fc    = lims.fc(1) + diff(lims.fc) * rand;
            burst_info.B     = lims.B(1) + diff(lims.B) * rand;
            % T_on in samples: convert from sample counts directly
            burst_info.T_on_mean = randi(lims.T_on_mean);
            burst_info.T_on_std  = randi(lims.T_on_std);
            burst_info.mu_ln     = lims.mu_ln(1) + diff(lims.mu_ln) * rand;
            burst_info.sigma_ln  = lims.sigma_ln(1) + diff(lims.sigma_ln) * rand;
            burst_info.M_max     = randi(lims.M_max);

            params.lims = lims;
            params.burst_info = burst_info;

            params.units.A          = char("linear");
            params.units.fc         = char("Hz");
            params.units.B          = char("Hz");
            params.units.T_on_mean  = char("samples");
            params.units.T_on_std   = char("samples");
            params.units.mu_ln      = char("log seconds");
            params.units.sigma_ln   = char("ratio");
            params.units.M_max      = char("integer");            
            
        case 13  % Phase-Coded Pulse Jammer (Barker-13 or P4 polyphase)
            % Constraints (tightened to guarantee multiple pulses per window
            % and avoid clipping pathology from sparse pulse trains)
            lims.A          = [0.8, 1.2];
            lims.fc         = [0.5e6, 4.5e6];
            lims.phi_range  = [0, 2*pi];
            lims.code_type  = {'barker13', 'p4'};
            lims.T_pulse    = [10e-6, 30e-6];        % was [5e-6, 50e-6]: bounded
            lims.PRF        = [50e3, 200e3];         % was [10e3, 100e3]: denser train
            lims.M          = [3, 10];               % was [1, 8]: at least 3 pulses

            % Sample core
            params.A   = lims.A(1) + diff(lims.A) * rand;
            params.fc  = lims.fc(1) + diff(lims.fc) * rand;
            params.phi = lims.phi_range(1) + diff(lims.phi_range) * rand;

            % Code selection
            code_idx = randi(2);
            pulse_info.code_type = char(lims.code_type{code_idx});

            if strcmp(pulse_info.code_type, 'barker13')
                pulse_info.N_chips = 13;
                barker_seq = [1,1,1,1,1,-1,-1,1,1,-1,1,-1,1];
                pulse_info.code_phase = (barker_seq(:) == -1) * pi;   % column, 0 or pi
            else  % p4 (true P4 polyphase)
                n_choices = [16, 32, 64];
                pulse_info.N_chips = n_choices(randi(3));
                k = (1:pulse_info.N_chips)';
                % True P4 formula: phi_k = (pi/N) * (k-1) * (k - 1 - N)
                pulse_info.code_phase = (pi / pulse_info.N_chips) * ...
                                (k - 1) .* (k - 1 - pulse_info.N_chips);
            end

            % Sample pulse duration, then derive integer chip length to keep
            % L_pulse = N_chips * L_chip exactly (avoids drift in the synthesizer)
            T_pulse_req = lims.T_pulse(1) + diff(lims.T_pulse) * rand;
            L_chip = max(1, round(T_pulse_req * spec.fs / pulse_info.N_chips));
            pulse_info.L_chip = L_chip;
            pulse_info.L_pulse = pulse_info.N_chips * L_chip;
            pulse_info.T_pulse = pulse_info.L_pulse / spec.fs;   % actual, not requested

            pulse_info.PRF = lims.PRF(1) + diff(lims.PRF) * rand;
            pulse_info.M   = randi(lims.M);

            % Place first pulse so the train is centered in the window
            T_obs = double(spec.N) / spec.fs;
            train_dur = (pulse_info.M - 1) / pulse_info.PRF + pulse_info.T_pulse;
            if train_dur < T_obs
                pulse_info.t_start = (T_obs - train_dur) * rand;
            else
                pulse_info.t_start = 0;
            end

            params.lims = lims;
            params.pulse_info = pulse_info;

            params.units.A         = char("linear");
            params.units.fc        = char("Hz");
            params.units.phi       = char("rad");
            params.units.code_type = char("string");
            params.units.N_chips   = char("integer");
            params.units.code_phase= char("rad vector");
            params.units.L_chip    = char("samples");
            params.units.L_pulse   = char("samples");
            params.units.T_pulse   = char("seconds (realized)");
            params.units.PRF       = char("Hz");
            params.units.M         = char("integer");
            params.units.t_start   = char("seconds");
            
        case 14  % Triangular FM (TFM) - Unknown Class 5
            % Constraints
            lims.A          = [0.8, 1.2];
            lims.delta_f    = [1e6, 4e6];       % Peak-to-peak sweep bandwidth (Hz)
            lims.fm       = [10e3, 50e3];     % Sweep repetition frequency (Hz)
            lims.K    = [0.3, 0.7];       % Hardware-induced sweep asymmetry
            
            % Sample basic parameters
            params.A        = lims.A(1) + diff(lims.A) * rand(1);
            params.delta_f  = lims.delta_f(1) + diff(lims.delta_f) * rand(1);
            params.fm       = lims.fm(1) + diff(lims.fm) * rand(1);
            % Randomize duty cycle to simulate non-ideal VCO charging/discharging
            params.K = lims.K(1) + diff(lims.K) * rand(1);
            
            params.lims = lims;
            
            % Units
            params.units.A          = char("linear");
            params.units.delta_f    = char("Hz");
            params.units.fm         = char("Hz");
            params.units.K   = char("ratio");

        case 15  % Direct Sequence Spread Spectrum (DSSS) / Unknown Class 6
            % Constraints
            lims.A      = [0.8, 1.2];
            lims.beta   = [0.2, 0.35];          % raised‑cosine roll‑off
            lims.Rc     = [1e6, 3e6];           % chip rate (Hz)
            lims.fc_range   = [0.5e6, 4.5e6];   % allowed carrier range
            lims.phi_range  = [0, 2*pi];

            params.A    = lims.A(1) + diff(lims.A) * rand(1);
            params.beta = lims.beta(1) + diff(lims.beta) * rand(1);
            params.Rc   = lims.Rc(1) + diff(lims.Rc) * rand(1);

            % Occupied bandwidth ≈ (1+beta)*Rc
            B_occ = (1 + params.beta) * params.Rc;

            % Carrier frequency – keep whole signal inside Nyquist band
            fc_min_valid = lims.fc_range(1) + B_occ/2;
            fc_max_valid = lims.fc_range(2) - B_occ/2;
            if fc_min_valid >= fc_max_valid
                params.fc = (lims.fc_range(1) + lims.fc_range(2)) / 2;
            else
                params.fc = fc_min_valid + (fc_max_valid - fc_min_valid) * rand(1);
            end

            params.phi = lims.phi_range(1) + diff(lims.phi_range) * rand(1);

            params.lims = lims;

            params.units.A     = char("linear");
            params.units.beta  = char("ratio");
            params.units.Rc    = char("Hz");
            params.units.fc    = char("Hz");
            params.units.phi   = char("rad");
 
        case 16  % Pulsed Gaussian Noise Jamming (PGNJ) – Unknown Class 7
            % Constraints
            lims.A          = [0.8, 1.2];
            lims.PRF        = [5e3, 50e3];     % pulse repetition frequency (Hz)
            lims.duty_cycle = [0.10, 0.40];    % low duty (distinct from PBNJ)
            lims.rise_samp  = [4, 16];         % envelope rise/fall length (samples)
 
            params.A          = lims.A(1) + diff(lims.A) * rand(1);
            pgn_info.PRF        = lims.PRF(1) + diff(lims.PRF) * rand(1);
            pgn_info.duty_cycle = lims.duty_cycle(1) + diff(lims.duty_cycle) * rand(1);
            pgn_info.rise_samp  = round(lims.rise_samp(1) + diff(lims.rise_samp) * rand(1));
 
            params.lims = lims;
            params.pgn_info = pgn_info;
 
            params.units.A          = char("linear");
            params.units.PRF        = char("Hz");
            params.units.duty_cycle = char("ratio");
            params.units.rise_samp  = char("samples");

        case 17 % Moving-Band Noise (MBN) / Unknown Class 8
            % Constraints
            lims.A              = [0.8, 1.2];
            lims.B              = [0.5e6, 0.8e6];       % Bandwidth (Hz)
            lims.delta_f_sweep  = [1.0e6, 1.8e6];       % Total sweep excursion (Hz)
            
            % Fixed parameters for Overlap-Add
            mbn_info.frame_len  = 256;
            mbn_info.hop_len    = 128;
            
            % Sample basic parameters
            params.A = lims.A(1) + diff(lims.A) * rand(1);
            mbn_info.B = lims.B(1) + diff(lims.B) * rand(1);
            mbn_info.delta_f_sweep = lims.delta_f_sweep(1) + diff(lims.delta_f_sweep) * rand(1);
            
            % Calculate valid start frequency to stay within Nyquist (0.5 to 4.5 MHz)
            f_min_allowed = 0.5e6 + mbn_info.B / 2;
            f_max_allowed = 4.5e6 - mbn_info.B / 2 - mbn_info.delta_f_sweep;
            
            mbn_info.f_start = f_min_allowed + (f_max_allowed - f_min_allowed) * rand(1);
            
            params.mbn_info = mbn_info;
            params.lims = lims;
            
            % Units
            params.units.A             = char("linear");
            params.units.B             = char("Hz");
            params.units.delta_f_sweep = char("Hz");
            params.units.f_start       = char("Hz");
            params.units.frame_len     = char("samples");
            params.units.hop_len       = char("samples");

        otherwise
            error('Invalid class_id: %d. Must be 0-17.', class_id);
    end

    required_fields = clean.get_active_fields(class_id);
    clean.validate_active_fields(params, required_fields);

    % Restore RNG State so as not to affect the rest of the workspace.
    rng(old_state);
end 

