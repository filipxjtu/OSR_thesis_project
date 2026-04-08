function generate_stat_report_clean(dataset)
    

    % Basic validation
    assert(isstruct(dataset), 'Input must be a dataset struct.');
    assert(isfield(dataset,'X_clean'), 'Missing X_clean.');
    assert(isfield(dataset,'y'), 'Missing y.');
    assert(isfield(dataset,'params'), 'Missing params.');
    assert(isfield(dataset,'meta'), 'Missing meta.');

    X = dataset.X_clean;
    y = dataset.y;
    params = dataset.params;
    meta = dataset.meta;

    classes = meta.class_set(:);
    n_classes = numel(classes);
   
    % Dataset size
    n_per_class = meta.n_per_class;
    total_samples = meta.Ns;
    N = meta.N;
    fs = meta.fs;
    seed = meta.dataset_seed;
    spec_version = meta.spec_version;


    % Generate report using the report generator function
    report = clean.report_clean_dataset_v2(dataset);

    % Prepare file
    filename = sprintf( ...
        'clean_dataset_%s_seed%d_n%d_statistical_report.md', ...
        spec_version, seed, n_per_class);
    
    intended_dir = fullfile('reports','statistical');
    if ~exist(intended_dir, 'dir')
        mkdir(intended_dir);
    end
    fid = fopen(fullfile(intended_dir, filename), 'w');

    fprintf(fid, '# Clean Dataset Artifact Report (Mathematical Summary)\n');
    fprintf(fid, 'Version: %s  \n', spec_version);
    fprintf(fid, 'Artifact: clean_dataset_%s_seed%d_n%d.mat  \n\n', ...
        spec_version, seed, n_per_class);
    
    % Specification Section
    fprintf(fid, '---\n\n');
    fprintf(fid, '## 1. Specification Used\n\n');
    fprintf(fid, '| Field | Value |\n');
    fprintf(fid, '|-------|-------|\n');
    fprintf(fid, '| spec_version | %s |\n', spec_version);
    fprintf(fid, '| fs | %.0f Hz |\n', fs);
    fprintf(fid, '| N | %d samples |\n', N);
    fprintf(fid, '| dataset_seed | %d |\n\n', seed);

    % Dataset Size
    fprintf(fid, '---\n\n');
    fprintf(fid, '## 2. Dataset Size\n\n');
    fprintf(fid, '| Quantity | Value |\n');
    fprintf(fid, '|----------|-------|\n');
    fprintf(fid, '| Classes | %d |\n', n_classes);
    fprintf(fid, '| n_per_class | %d |\n', n_per_class);
    fprintf(fid, '| total_samples | %d |\n', total_samples);
    fprintf(fid, '| Matrix shape (X_clean) | %d × %d |\n\n', N, total_samples);

    % Parameter Min/Max per Class
    fprintf(fid, '---\n\n');
    fprintf(fid, '## 3. Realized Parameter Ranges (Per Class)\n\n');
    fprintf(fid, '| Class | Parameter | Min | Max |\n');
    fprintf(fid, '|-------|-----------|-----|-----|\n');

    for c_idx = 1:n_classes
        c = classes(c_idx);

        idx = (y == c);
        class_params = params(idx);

        if isempty(class_params)
            continue;
        end

        param_fields = fieldnames(params(1));
        assert(~isempty(param_fields), 'Parameter schema is empty.');

        for f = 1:numel(param_fields)
            field_name = param_fields{f};

            % Only numeric scalar fields
            values = [];

            for k = 1:numel(class_params)
                if isfield(class_params(k), field_name)
                    val = class_params(k).(field_name);
            
                    if isnumeric(val) && isscalar(val)
                        values(end+1) = val; %#ok<AGROW>
                    end
                end
            end

            if ~isempty(values)
                fprintf(fid, '| %d | %s | %.6g | %.6g |\n', ...
                    c, field_name, min(values), max(values));
            end
        end
    end


    % Per-Class Statistics
    fprintf(fid, '\n---\n\n');
    fprintf(fid, '## 4. Per-Class Signal Statistics\n\n');
    fprintf(fid, '| Class | Mean | Std | RMS | Checksum |\n');
    fprintf(fid, '|-------|------|-----|-----|----------|\n');

    for c_idx = 1:n_classes
        c = classes(c_idx);

        row_idx = report.per_class_stats.class_id == c;
        
        if any(row_idx)
            mean_val = report.per_class_stats.mean(row_idx);
            std_val  = report.per_class_stats.std(row_idx);
            rms_val  = report.per_class_stats.rms(row_idx);

            idx = (y == c);
            Xk = X(:, idx);
            checksum_val = sum(abs(Xk(:)));

            fprintf(fid, '| %d | %.6g | %.6g | %.6g | %.6g |\n', ...
                c, mean_val, std_val, rms_val, checksum_val);
        end
    end

    fprintf(fid, '\n---\n\n');
    fprintf(fid, '## 5. Dataset Integrity Evidence\n\n');
    fprintf(fid, 'Global checksum:\n\n');
    fprintf(fid, '```\n%.12g\n```\n', report.global_checksum);

    fclose(fid);
    fprintf('Statistical report generated: %s\n', filename);

end