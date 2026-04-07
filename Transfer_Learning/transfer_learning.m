% 1. データセットの準備
% MathWorks Merchデータセットを解凍します。このデータセットには、5つの異なるクラスに属する75個のイメージが含まれています。
folderName = "MerchData";
if ~exist(folderName, "dir")
    unzip("MerchData.zip", folderName);
end
% イメージデータストアを作成します。サブフォルダー名がイメージラベルに対応していることを示します。
imds = imageDatastore(folderName, ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames"); %

% サンプルイメージを表示します（オプション）。
% numImages = numel(imds.Labels);
% idx = randperm(numImages, 16);
% I = imtile(imds, Frames=idx);
% figure;
% imshow(I);
% title('Sample Images from MerchData'); %

% クラス名とクラス数を抽出します。
classNames = categories(imds.Labels); %
numClasses = numel(classNames); %
fprintf('分類するクラスの数: %d\n', numClasses);
disp('クラス名:');
disp(classNames);

% データを学習用、検証用、テスト用に分割します。
% イメージの70%を学習に、15%を検証に、15%をテストに使用します。
[imdsTrain, imdsValidation, imdsTest] = splitEachLabel(imds, 0.7, 0.15, 0.15, "randomized"); %
fprintf('学習イメージの数: %d\n', numel(imdsTrain.Files));
fprintf('検証イメージの数: %d\n', numel(imdsValidation.Files));
fprintf('テストイメージの数: %d\n', numel(imdsTest.Files));

% 2. 学習済みネットワークの読み込みと変更
% 事前学習済みのResNet-18ネットワークを読み込みます。
% ResNet-18は、100万個を超えるイメージで学習されており、1000個のオブジェクトカテゴリに分類できます。
net = imagePretrainedNetwork("resnet18"); %

% ネットワークアーキテクチャを表示して、入力層を確認します。
% analyzeNetwork(net); %
inputSize = net.Layers(1).InputSize; %
fprintf('ネットワークの入力サイズ: [%d %d %d]\n', inputSize);

% 転移学習のためにネットワークの最後の層を変更します。
% 既存の分類層を新しいタスクのクラス数に合うように置き換える必要があります。
% ResNet-18の場合、通常、'fc1000'や'prob'のような最終の全結合層と分類層を置き換えます。
% dlnetworkオブジェクトとしてネットワークを取得し、層を変更します。
lgraph = layerGraph(net); %

% 最後の分類層を削除します。
lgraph = removeLayers(lgraph, lgraph.Layers(end).Name); %

% 新しいタスクに合わせて、新しい全結合層と分類層を追加します。
% 新しい全結合層は、numClassesで定義された新しいクラス数に対応します。
newLayers = [
    fullyConnectedLayer(numClasses, 'Name', 'fc_new') %
    softmaxLayer('Name', 'softmax_new')]; %

% ネットワークの最終層（'fc1000_softmax'）の前の層名を見つけます。
% ResNet-18では、通常、分類層の前に'prob'層があります。
% 'fc1000'層を削除し、新しい層を接続します。
lgraph = removeLayers(lgraph, 'fc1000');
lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'pool5', 'fc_new'); % 'pool5'はResNet-18の最後のプーリング層の出力名です。

% dlnetworkオブジェクトとしてネットワークを再構築します。
net = dlnetwork(lgraph); %

% 3. データ拡張と学習オプションの指定
% ネットワークの入力サイズに合わせて学習イメージを自動的にサイズ変更するためのaugmentedImageDatastoreを作成します。
% データ拡張は、過適合を防止し、ネットワークの汎化能力を向上させるのに役立ちます。
pixelRange = [-4 4]; % ピクセルのランダム平行移動
imageAugmenter = imageDataAugmenter( ...
    RandXReflection=true, ... % 水平反転
    RandXTranslation=pixelRange, ... % X方向の平行移動
    RandYTranslation=pixelRange); % Y方向の平行移動

augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    DataAugmentation=imageAugmenter); %
augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation); %
augimdsTest = augmentedImageDatastore(inputSize(1:2), imdsTest); %

% 学習オプションを指定します。
% 'adam'ソルバーを使用し、ミニバッチサイズ、エポック数などを設定します。
options = trainingOptions('adam', ... %
    InitialLearnRate=0.0001, ... % 初期学習率
    MaxEpochs=5, ... % 最大エポック数
    MiniBatchSize=10, ... % ミニバッチサイズ
    Shuffle='every-epoch', ... % 各エポックでデータをシャッフル
    ValidationData=augimdsValidation, ... % 検証データ
    ValidationFrequency=floor(numel(imdsTrain.Files)/10), ... % 検証の頻度
    Plots='training-progress', ... % 学習進捗プロットを表示
    Verbose=false); % コマンドライン出力を非表示

% 4. ネットワークの学習
% trainnet関数を使用してネットワークに学習させます。
% クロスエントロピー損失関数を使用します。
fprintf('ネットワークの学習を開始します...\n');
trainedNet = trainnet(augimdsTrain, net, 'crossentropy', options); %
fprintf('ネットワークの学習が完了しました。\n');

% 5. ネットワークのテスト
% テストデータを使用して、ネットワークの性能を評価します。
% minibatchpredict関数を使用して予測を行います。
YPred = minibatchpredict(trainedNet, augimdsTest); %
% 予測スコアをラベルに変換します。
YPred = scores2label(YPred, classNames); %

% 真のラベルを取得します。
TTest = imdsTest.Labels;

% 分類精度を計算します。精度は、正しい予測の割合です。
accuracy = mean(YPred == TTest); %
fprintf('テストセットの分類精度: %.2f%%\n', accuracy * 100);

% 混同チャートで予測を可視化します。
figure;
confusionchart(TTest, YPred); %
title('テストデータの混同行列');

% 6. 新しいデータでの予測の実行
% 新しい単一イメージを分類します。
% 例として、テストセットから最初の画像を使用します。
newImageFile = imdsTest.Files{1};
im = imread(newImageFile);

% ネットワークの入力サイズに合わせてイメージのサイズを変更します。
im = imresize(im, inputSize(1:2));

% singleデータ型に変換します。
X = single(im);

% GPUが利用可能な場合は、データをgpuArrayに変換します（予測を高速化するため）。
if canUseGPU
    X = gpuArray(X);
end

% predict関数を使用して予測を行います。
scores = predict(trainedNet, X); %
[label, score] = scores2label(scores, classNames); %

% 予測されたラベルと対応するスコアを含むイメージを表示します。
figure;
imshow(im);
title(sprintf('予測: %s (スコア: %.2f)', string(label), gather(score))); %