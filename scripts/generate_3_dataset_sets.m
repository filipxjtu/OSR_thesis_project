clear
clc

s1 = 32;
s2 = 54;
s3 = 76;
n = 600;

addpath('C:\Users\user\Documents\MATLAB\thesis_project\matlab')
addpath('C:\Users\user\Documents\MATLAB\thesis_project\scripts')

spec = core.get_canonical_spec();

clean1 = run_clean_pipeline(spec, n, s1);
fprintf('\n');
clean2 = run_clean_pipeline(spec, n, s2);
fprintf('\n');
clean3 = run_clean_pipeline(spec, n, s3);
fprintf('\n');
train1 = run_impaired_pipeline(spec, n, s1, 'train');
fprintf('\n');
train2 = run_impaired_pipeline(spec, n, s2, 'train');
fprintf('\n');
train3 = run_impaired_pipeline(spec, n, s3, 'train');
fprintf('\n');
eval1 = run_impaired_pipeline(spec, n, s1, 'eval');
fprintf('\n');
eval2 = run_impaired_pipeline(spec, n, s2, 'eval');
fprintf('\n');
eval3 = run_impaired_pipeline(spec, n, s3, 'eval');
fprintf('\n');
unknown1 = run_unknown_pipeline(spec, n, s1);
fprintf('\n');
unknown2 = run_unknown_pipeline(spec, n, s2);
fprintf('\n');
unknown3 = run_unknown_pipeline(spec, n, s3);