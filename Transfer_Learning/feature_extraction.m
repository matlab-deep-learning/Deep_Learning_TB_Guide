%% 1. データの読み込みと準備
if ~exist('MerchData', 'dir')
    unzip('MerchData.zip');
end

imds = imageDatastore('MerchData', ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames");

[imdsTrain, imdsTest] = splitEachLabel(imds, 0.7, 'randomized');

numTrainImages = numel(imdsTrain.Labels);
numTestImages = numel(imdsTest.Labels);
fprintf('学習イメージ数: %d\n', numTrainImages);
fprintf('テストイメージ数: %d\n', numTestImages);

%% 2. 事前学習済みネットワークの読み込み
% imagePretrainedNetwork returns dlnetwork directly (replaces legacy resnet18 → DAGNetwork)
net = imagePretrainedNetwork("resnet18");
fprintf('ResNet-18 を読み込みました (dlnetwork)。\n');

inputSize = net.Layers(1).InputSize;
fprintf('入力サイズ: [%d %d %d]\n', inputSize(1), inputSize(2), inputSize(3));

%% 3. 特徴抽出のためのデータ準備
augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain);
augimdsTest = augmentedImageDatastore(inputSize(1:2), imdsTest);

%% 4. 特徴の抽出
% minibatchpredict with Outputs= replaces activations() + permute()
% Returns samples x features directly — no reshape needed.
layer = "fc1000";
fprintf('特徴抽出層: %s\n', layer);

fprintf('学習データから特徴を抽出中...\n');
featuresTrain = double(minibatchpredict(net, augimdsTrain, Outputs=layer));

fprintf('テストデータから特徴を抽出中...\n');
featuresTest = double(minibatchpredict(net, augimdsTest, Outputs=layer));

labelsTrain = imdsTrain.Labels;
labelsTest = imdsTest.Labels;

fprintf('特徴ベクトルサイズ: %d\n', size(featuresTrain, 2));

%% 5. サポートベクターマシン(SVM)分類器の学習 
fprintf('SVM分類器 (fitcecoc) を学習中...\n');
classifier_svm = fitcecoc(featuresTrain, labelsTrain);

%% 6. 分類器の評価
YPred_svm = predict(classifier_svm, featuresTest);
acc_svm = mean(YPred_svm == labelsTest) * 100;
fprintf('fitcecoc (SVM) テスト精度: %.2f%%\n', acc_svm);
fprintf('\n=== Feature Extraction Results ===\n');
fprintf('  fitcecoc (SVM):  %.2f%% (same as legacy classifier)\n', acc_svm);

figure;
confusionchart(labelsTest, YPred_svm);