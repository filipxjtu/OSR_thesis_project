clear
clc

addpath('C:\Users\user\Documents\MATLAB\thesis_project\matlab')
addpath('C:\Users\user\Documents\MATLAB\thesis_project\scripts')

spec = core.get_canonical_spec();

clean1 = run_clean_pipeline(spec, 600, 17);
fprintf('\n');
clean2 = run_clean_pipeline(spec, 600, 27);
fprintf('\n');
clean3 = run_clean_pipeline(spec, 600, 37);
fprintf('\n');
train1 = run_impaired_pipeline(spec, 600, 17, 'train');
fprintf('\n');
train2 = run_impaired_pipeline(spec, 600, 27, 'train');
fprintf('\n');
train3 = run_impaired_pipeline(spec, 600, 37, 'train');
fprintf('\n');
eval1 = run_impaired_pipeline(spec, 600, 17, 'eval');
fprintf('\n');
eval2 = run_impaired_pipeline(spec, 600, 27, 'eval');
fprintf('\n');
eval3 = run_impaired_pipeline(spec, 600, 37, 'eval');