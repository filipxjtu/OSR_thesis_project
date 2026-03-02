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
            lims.A   = [0.8, 1.2];          
            lims.f0  = [1e6, 4e6];  
            lims.phi = [0, 2*pi];           
            
            % Sample
            params.A   = lims.A(1)   + diff(lims.A)   * rand(1);
            params.f0  = lims.f0(1)  + diff(lims.f0)  * rand(1);
            params.phi = lims.phi(1) + diff(lims.phi) * rand(1);

            params.lims = lims;
            
            % Units
            params.units.A   = char("linear");
            params.units.f0  = char("Hz");
            params.units.phi = char("rad");

        case 1  % Multi-Tone (Clean)
            % Constraints
            lims.K_range = [3, 5];             % small and bounded
            lims.A_total_max = 1.5;            % avoids clipping
            lims.f0 = [0.5e6, 4.5e6];      % Within fs/2 (10MHz fs)
            lims.phi = [0, 2*pi]; 
            
            % sample K (Number of tones)
            params.K = randi(lims.K_range);
            
            % sample amplitudes 
            raw_A = rand(params.K, 1);             % column vector
            params.A = (raw_A / sum(raw_A)) * lims.A_total_max;        % scaled to avoid clipping
            
            % sample frequencies
            params.f0 = sort(lims.f0(1) + diff(lims.f0) * rand(params.K, 1));
            
            % sample phases
            params.phi = lims.phi(1) + diff(lims.phi) * rand(params.K, 1);
            
            params.lims = lims;

            % Units
            params.units.A = char("linear vector");
            params.units.f0 =  char("Hz vector");
            params.units.phi = char("rad vector");
            params.units.K = char("integer");

        case 2  % LFM sweep (Clean)
            % Constraints
            lims.A   = [0.8, 1.2];          
            f_min = 0.1 * (spec.fs /2);
            f_max = 0.9 * (spec.fs /2); 
            lims.f = [f_min, f_max];
            lims.phi = [0, 2*pi];
            
            % Sample
            params.A   = lims.A(1)   + diff(lims.A)   * rand(1);
            f_pair = sort(lims.f(1) + diff(lims.f) * rand(2, 1));
            params.f0 = f_pair(1);
            params.f1 = f_pair(2);
            params.phi = lims.phi(1) + diff(lims.phi) * rand(1);
            params.T = double(spec.N) / double(spec.fs);

            params.lims = lims;
            
            % Units
            params.units.A   = char("linear");
            params.units.f0  = char("Hz");
            params.units.f1  = char("Hz");
            params.units.phi = char("rad");
            params.units.T = char("seconds");
        
          case 3  % SFM sweep (Clean)
            % Constraints
            lims.A   = [0.8, 1.2];
            lims.fc = [0.3 * (spec.fs /2), 0.7 * (spec.fs/2)];
            lims.df = [0.02 * (spec.fs /2), 0.15 * (spec.fs/2)];
            lims.fm_ratio = [0.01, 0.05];        % 1% to 10% of fc
            lims.phi = [0, 2*pi];
            
            % Sample
            params.A   = lims.A(1)   + diff(lims.A)   * rand(1);         
            params.fc = lims.fc(1) + diff(lims.fc) * rand(1);
            params.df = lims.df(1) + diff(lims.df) * rand(1);
            ratio = lims.fm_ratio(1) + diff(lims.fm_ratio) * rand(1);
            params.fm = ratio * params.fc;
            params.phi = lims.phi(1) + diff(lims.phi) * rand(1);
            assert(params.fm > 0,'SFM, fm should be positive');
            params.beta = params.df / params.fm;
            assert(params.fc + params.df < spec.fs/2, 'SFM violates Nyquist constraint.');
            
            params.lims = lims;

            % Units
            params.units.A   = char("linear");
            params.units.fc  = char("Hz");
            params.units.df  = char("Hz");
            params.units.fm  = char("Hz");
            params.units.phi = char("rad");
            params.units.beta = char("ratio");
        
        case 4  % PBN (Clean)
            % constants and FFT Setup
            nfft = spec.N;
            freq_res = spec.fs / nfft;
            nyquist = spec.fs / 2;
            
            % constraints
            lims.A  = [0.8, 1.2];
            lims.fL = [0.05 * nyquist, 0.6 * nyquist]; % 5% to 60% of Nyquist
            lims.BW = [0.05 * nyquist, 0.25 * nyquist]; % 5% and 25% of Nyquist
            lims.phi = [0, 2*pi];
            
            % amplitude scale
            params.A = lims.A(1) + diff(lims.A) * rand(1);

            % sample frequency band
            params.fL = lims.fL(1) + diff(lims.fL) * rand(1);
            params.bandwidth  = lims.BW(1) + diff(lims.BW) * rand(1);
            params.fH = params.fL + params.bandwidth;
            
            % convert to bin indices [bin 1 is DC (0 Hz)] 
            bin_L = ceil(params.fL / freq_res) + 1;
            bin_H = floor(params.fH / freq_res) + 1;

            % Nyquist bin index (for even nfft)
            nyq_bin = floor(nfft/2) + 1;

            % assertions
            assert(bin_L >= 2, 'PBN: bin_L must be >= 2 (avoid DC).');
            assert(bin_H <= nyq_bin - 1, 'PBN: bin_H must be < Nyquist bin.');
            assert(bin_H >= bin_L, 'PBN: invalid band bins.');

            % sample phases for bins in band
            n_band_bins = bin_H - bin_L + 1;
            phi_bins = lims.phi(1) + diff(lims.phi) * rand(n_band_bins, 1);

            params.lims = lims;

            params.bin_info.nfft = nfft;
            params.bin_info.freq_res = freq_res;
            params.bin_info.bin_L = bin_L;
            params.bin_info.bin_H = bin_H;
            params.bin_info.phi_bins = phi_bins;
            
            % Units
            params.units.A      = char("linear");
            params.units.nfft  = char("samples");
            params.units.fL     = char("Hz");
            params.units.fH     = char("Hz");
            params.units.bin_L  = char("index");
            params.units.bin_H  = char("index");
            params.units.phi_bins  = char("rad vector");
            params.units.freq_res = char("Hz");
            params.units.bandwidth = char("Hz");
            
        case 5 % Noise FM (Clean)
            % setup and grid
            nfft = spec.N;
            mod_freq_res = spec.fs / nfft;
            nyquist = spec.fs / 2;
            nyq_bin = floor(nfft / 2) + 1;
            
            % constraints
            lims.A     = [0.8, 1.2];
            lims.fc    = [0.2 * nyquist, 0.6 * nyquist];   % Carrier range
            lims.fL    = [0.02 * nyquist, 0.2 * nyquist];  % Modulator band floor
            lims.BW    = [0.02 * nyquist, 0.15 * nyquist]; % Modulator bandwidth
            lims.kappa = [5, 50];                          % Modulation sensitivity
            lims.phi   = [0, 2*pi];
            
            % sample scalar parameters
            params.A     = lims.A(1) + diff(lims.A) * rand(1);
            params.fc    = lims.fc(1) + diff(lims.fc) * rand(1);
            params.kappa = lims.kappa(1) + diff(lims.kappa) * rand(1);
            params.phi  = lims.phi(1) + diff(lims.phi) * rand(1);
            
            % sample Modulator Frequency Band
            fL_mod = lims.fL(1) + diff(lims.fL) * rand(1);
            BW_mod = lims.BW(1) + diff(lims.BW) * rand(1);
            fH_mod = fL_mod + BW_mod;
            
            % convert to bin indices
            mod_bin_L = ceil(fL_mod / mod_freq_res) + 1;
            mod_bin_H = floor(fH_mod / mod_freq_res) + 1;
            
            % assertions
            assert(mod_bin_L >= 2, 'NoiseFM: bin_L must avoid DC.');
            assert(mod_bin_H <= nyq_bin - 1, 'NoiseFM: bin_H must stay below Nyquist.');
            assert(mod_bin_H >= mod_bin_L, 'NoiseFM: invalid modulator band.');
            
            % sample Modulator Phases
            n_band_bins = mod_bin_H - mod_bin_L + 1;
            phi_bins = lims.phi(1) + diff(lims.phi) * rand(n_band_bins, 1);
            
            params.fL = fL_mod;
            params.fH = fH_mod;
            params.bandwidth = BW_mod;
            params.lims = lims;
            
            params.bin_info.nfft = nfft;
            params.bin_info.freq_res = mod_freq_res;
            params.bin_info.bin_L = mod_bin_L;
            params.bin_info.bin_H = mod_bin_H;
            params.bin_info.phi_bins = phi_bins;

            % Units
            params.units.A        = char("linear");
            params.units.fc       = char("Hz");
            params.units.kappa    = char("rad/s");
            params.units.phi     = char("rad");
            params.units.phi_bins = char("rad vector");
            params.units.bin_L    = char("index");
            params.units.bin_H    = char("index");
            params.units.nfft     = char("samples");
            params.units.freq_res = char("Hz");
            params.units.bandwidth = char("Hz");

        case 6  % Frequency Hopping (FH)(Clean)
            % constants
            Lhop_options = [128, 256, 512]; 
            grid_freq = 16;   
            
            % constraints
            lims.A    = [0.8, 1.2];
            lims.f_range   = [0.1 * (spec.fs /2) , 0.9 * (spec.fs / 2 )]; 
            lims.phi = [0, 2*pi];
            
            % sampling basic parameters
            params.A    = lims.A(1) + diff(lims.A) * rand(1);
            hop_info.Lhop = Lhop_options(randi(numel(Lhop_options)));
           
            % deterministic frequency grid (hop_set)
            hop_info.hop_set = linspace(lims.f_range(1), lims.f_range(2), grid_freq)';
            assert(max(hop_info.hop_set) < spec.fs/2, 'FH grid violates Nyquist constraint.');

            % determine num of hops and sample indices
            hop_info.n_hops = ceil(double(spec.N) / double(hop_info.Lhop));
            
            % randomly pick - result is a vector of integers from 1 to 16
            hop_info.hop_idx = randi(numel(hop_info.hop_set), [hop_info.n_hops, 1]);

            % random phase hops
            hop_info.phi_hops = lims.phi(1) + diff(lims.phi) * rand(hop_info.n_hops,1);
            
            params.hop_info = hop_info;
            params.lims = lims;

            % Units
            params.units.A       = char("linear");
            params.units.Lhop    = char("samples");
            params.units.hop_set = char("Hz vector (grid)");
            params.units.hop_idx = char("vector of indices");
            params.units.n_hops  =  char("integer");
            params.units.phi_hops = char("rad vector");

        otherwise
            error('Invalid class_id: %d. Must be 0-6.', class_id);
    end

    required_fields = clean.get_active_fields(class_id);
    clean.validate_active_fields(params, required_fields);

    % Restore RNG State so as not to affect the rest of the workspace.
    rng(old_state);
end 