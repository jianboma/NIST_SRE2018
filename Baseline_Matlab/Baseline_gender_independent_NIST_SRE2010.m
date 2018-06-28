
% clear all
% clc
tv_dim = 600; 
ubm = './mat_file/Gender_Ind_UBM_2048.mat';



%% step1: feature extraction




%% step2: UBM training
addpath SCRIPTS
% % 1. verify data location
filename = 'your_listfile_need_to_check';
filename_missing = './List_File/female_missing.scp';
dataloc = 'your_data_location';
% % 2. check if file exist
%background_data_existence(filename,dataloc,filename_missing);

% % 2. delete the filenames that are missing


% % 3. remove bad files and generate data for UBM training
filename = 'your_listfile_need_to_check';
filename_NaN = './List_File/female_NaN.scp';
dataloc = 'your_data_location';
UBM_data_location = 'your_data_for_training_ubm_location';
% NAN_zero_remove_data_UBM(filename,dataloc,filename_NaN,UBM_data_location)

% % 4. delete bad files

% % 5. create list file for UBM and training


% % 6. training

cd ./trainUBM
TrainUBM_Par;
cd ..

%% step3: statistics extraction and T matrix estimation

addpath ./MSRToolBox/code
nworkers = 16;

filename_Tmatrix_MSR = './List_File/Gender_Ind_T_Matrix_Training.scp';
Feature_path = 'you_feature_path';
% 1. create list file
% createlistMSR(filename1,Feature_path,filename2)

% 2: Learning the total variability subspace from background data

% % -----------------stage 1, compute BW statistics
ubm = './mat_file/Gender_Ind_UBM_2048.mat';
tv_dim = 600; 
niter  = 10;
fid = fopen(filename_Tmatrix_MSR, 'rt');
C = textscan(fid, '%s %s');
fclose(fid);
feaFiles = C{1};
stats = cell(length(feaFiles), 1);
dbstop if error
parfor file = 1 : length(feaFiles),
    display(num2str(file));
    [N, F] = compute_bw_stats(feaFiles{file}, ubm);
    stats{file} = [N; F];
end
save('./mat_file/background_stats_Gender_Ind.mat','stats','-v7.3');
% % ------------------------------------------------------%

% % -----------------stage 2, train T matrix
load('./mat_file/background_stats_Gender_Ind.mat');
T = train_tv_space(stats2, ubm, tv_dim, niter, nworkers);
save('./mat_file/T_Gender_Ind.mat','T','-v7.3');

%% step4:i-vectors Inference for background data and GPLDA hyper-parameters training
load('./mat_file/T_Gender_Ind.mat');
load('./mat_file/background_stats_Gender_Ind.mat');

lda_dim = 200;
nphi    = 200;
niter   = 10;
dataList = './List_File/Gender_Ind_T_Matrix_Training.scp';
fid = fopen(dataList, 'rt');
C = textscan(fid, '%s %s');
fclose(fid);
feaFiles = C{1};
dev_ivs = zeros(tv_dim, length(feaFiles));
parfor file = 1 : length(feaFiles),
    dev_ivs(:, file) = extract_ivector(stats{file}, ubm, T);
end
save('./mat_file/dev_ivs.mat','dev_ivs','-v7.3');

% reduce the dimensionality with LDA
spk_labs = C{2};
V = lda(dev_ivs, spk_labs);
save('./mat_file/V_matrix.mat','V');

dev_ivs = V(:, 1 : lda_dim)' * dev_ivs;
%------------------------------------
plda = gplda_em(dev_ivs, spk_labs, nphi, niter);
save('./mat_File/plda.mat','plda');

%% step5:i-vectors Inference for target and test data
%--------------------extract ivectors for targets----------------%
fid = fopen('./List_File/Gender_Ind_8conv-10sec_trainlParsed_MSR.scp', 'rt');
C = textscan(fid, '%s %s');
fclose(fid);

model_ids = unique(C{1,2});

spk_files = C{1,1};
model_files = spk_files;
nspks = length(unique(model_ids));
model_ivs = zeros(tv_dim, nspks);


%----for one segment enrolment data-----%
% parfor spk = 1 : nspks,
%     
%     [N, F] = compute_bw_stats(spk_files{spk}, ubm);
%     model_ivs(:, spk) = extract_ivector([N; F], ubm, T);
% %     disp(num2str(spk));
% end
% save('./Mat_File/Gender_Ind_8conv_10sec_malemodel.mat','model_ivs');
%-----------------for multiply segments enrolment data--------------------------%
parfor spk = 1 : nspks,
    ids = find(ismember(C{1,2}, model_ids{spk}));
    spk_files = model_files(ids);

    N = 0; F = 0; 
    for ix = 1 : length(spk_files),
        [n, f] = compute_bw_stats(spk_files{ix}, ubm);
        N = N + n; F = f + F; 
        model_ivs(:, spk) = model_ivs(:, spk) + extract_ivector([n; f], ubm, T);
    end
%     model_ivs2(:, spk) = extract_ivector([N; F]/length(spk_files), ubm, T); % stats averaging!
    model_ivs(:, spk) = model_ivs(:, spk)/length(spk_files); % i-vector averaging!
end

save('./mat_File/Gender_Ind_8conv_10sec_model.mat','model_ivs');
% % 
%----------------------------------------------------------------------------------%
%-------extract ivector for test files--------------------------------------%
fid = fopen('./List_File/Gender_Ind_8conv-10sec_testParsed_MSR.scp', 'rt');
C = textscan(fid,'%s %s');
fclose(fid);
test_labels = C{1,2};
test_files = C{1,1};
test_ivs = zeros(tv_dim, length(test_files));
parfor tst = 1 : length(test_files),
    [N, F] = compute_bw_stats(test_files{tst}, ubm);
    test_ivs(:, tst) = extract_ivector([N; F], ubm, T);
end
save('./mat_file/Gender_Ind_8conv_10sec_test.mat','test_ivs');



%% step6:evaluation

fid = fopen('./List_File/Gender_Ind_8conv-10sec_keyParsed_CC5_Scored.scp', 'rt');
C = textscan(fid, '%s %s %s %s %s %s');
fclose(fid);
[Model_labes,~,Kmodel] = unique(C{1,1});
[Test_labes,~,Ktest] = unique(C{1,2});
nspks = length(Model_labes);

lda_dim = 200;

load('./mat_File/plda.mat');
load('./mat_File/V-matrix.mat');
load('./mat_File/Gender_Ind_8conv_10sec_model.mat');
load('./mat_File/Gender_Ind_8conv_10sec_test.mat');
model_ivs = V(:, 1 : lda_dim)' * model_ivs;
test_ivs = V(:, 1 : lda_dim)' * test_ivs;

%------------------------------------%
scores = score_gplda_trials(plda, model_ivs, test_ivs);
linearInd =sub2ind([nspks, length(Test_labes)], Kmodel, Ktest);
scores = scores(linearInd); % select the valid trials

labels = C{1,3};

[eer,DCF08,DCF10] = compute_eer(scores, labels, true);
[eer,DCF08,DCF10] % eer and dcf values





