
function x_clean = synthesize_clean_signal_class18(params, spec)
% SYNTHESIZE_CLEAN_SIGNAL_CLASS6
% BPSK Smart Jammer with SRRC pulse shaping

    N  = double(spec.N);
    fs = double(spec.fs);

    SPS = params.bpsk_info.SPS;
    span = params.bpsk_info.filter_span;

    % required number of symbols
    filter_delay = span * SPS;
    K = ceil(N / SPS) + filter_delay;

    % generate BPSK symbols
    bits = randi([0,1], K, 1);
    b = 2*bits - 1;   % {-1, +1}

    % SRRC filter
    h = rcosdesign(params.alpha, span, SPS, 'sqrt');

    % pulse shaping (efficient)
    x_bb = upfirdn(b, h, SPS, 1);

    % remove filter delay
    delay = span * SPS / 2;
    x_bb = x_bb(delay+1:end);

    % truncate to N
    if numel(x_bb) < N
        x_bb = [x_bb; zeros(N - numel(x_bb),1)];
    else
        x_bb = x_bb(1:N);
    end

    % Upconversion
    t = (0:N-1)' / fs;
    carrier = exp(1i * (2*pi*params.fc*t + params.phi));

    x = x_bb .* carrier;

    % amplitude
    x = params.A * x;

    % normalize to unit RMS
    rms_val = sqrt(mean(abs(x).^2));
    assert(rms_val > 0, 'BPSK: RMS is zero before normalization.');
    x_clean = x / rms_val;

    % Assertions
    assert(iscolumn(x_clean), 'Output must be a column vector.');
    assert(numel(x_clean) == spec.N, 'Output length mismatch.');
    assert(~isreal(x_clean), 'Signal must be complex.');
    assert(~all(imag(x_clean(:)) == 0), ...
        'Signal must have non-zero imaginary component.');
    assert(all(isfinite(x_clean(:))), ...
        'Signal contains NaN/Inf.');
end