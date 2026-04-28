clear
clc

s1 = 410;
s2 = 118;
s3 = 276;
s4 = 314;
s5 = 152;
s6 = 340;
s7 = 142;
s8 = 264;
s9 = 336;
s10 = 608;
s11 = 530;
s12 = 472;
s13 = 214;

n = 500;

addpath('C:\Users\user\Documents\MATLAB\thesis_project\matlab')
addpath('C:\Users\user\Documents\MATLAB\thesis_project\scripts')

spec = core.get_canonical_spec();
x = 10;
for i = 1:13
    s = sprintf("s%d",i);
    v = eval(s);
    run_impaired_pipeline(spec, n, v, 'train', x);
    run_impaired_pipeline(spec,n, v, 'eval',x)
    run_unknown_pipeline(spec, n, v, x)
    fprintf('\n');
    x = x-2;
end


s1 = 38;
s2 = 55;
s3 = 83;
n = 2500;

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