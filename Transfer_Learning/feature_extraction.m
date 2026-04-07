% 1. データの読み込みと準備
% この例では、MathWorks™ Merch データセットを使用します。
% 'MerchData.zip' が現在のフォルダにない場合はダウンロードして解凍します。

unzip('MerchData.zip');

% imageDatastore を使用してイメージデータを読み込みます。
% フォルダ名に基づいて自動的にラベルが付けられます。
imds = imageDatastore('MerchData', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% データを学習データ (70%) とテストデータ (30%) に分割します。
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.7, 'randomized');

% 学習イメージと検証イメージの数を表示します。
numTrainImages = numel(imdsTrain.Labels);
numTestImages = numel(imdsTest.Labels);
fprintf('学習イメージ数: %d\n', numTrainImages);
fprintf('テストイメージ数: %d\n', numTestImages);

% 2. 事前学習済みResNet-18ネットワークの読み込み
% resnet18は、100万個を超えるイメージで学習済みであり、広範囲のイメージに対する豊富な特徴表現を学習しています。
% 初めて使用する場合、Deep Learning Toolbox Model for ResNet-18 Networkサポートパッケージのダウンロードが求められることがあります。
net = resnet18;
fprintf('事前学習済みネットワーク: Resnet18 を読み込みました。\n');

% ネットワークの入力層からイメージ入力サイズを取得します。
inputSize = net.Layers(1).InputSize;
fprintf('ネットワーク入力サイズ: [%d %d %d]\n', inputSize(1), inputSize(2), inputSize(3));

% 3. 特徴抽出のためのデータの準備
% activations関数への入力として、ネットワークの入力サイズに合わせてイメージのサイズを自動的に変更する
% augmentedImageDatastoreを作成します。
augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain);
augimdsTest = augmentedImageDatastore(inputSize(1:2), imdsTest);

% 4. 特徴の抽出
% ネットワークから特徴を抽出する層を指定します。
% 'fc1000'層は、特徴を抽出するために使用されます。
layer = 'fc1000';
fprintf('特徴を抽出する層: %s\n', layer);

% 学習データから特徴を抽出します。
fprintf('学習データから特徴を抽出中...\n');
featuresTrain = activations(net, augimdsTrain, layer);
featuresTrain = permute(featuresTrain,[4,3,1,2]);

% テストデータから特徴を抽出します。
fprintf('テストデータから特徴を抽出中...\n');
featuresTest = activations(net, augimdsTest, layer);
featuresTest = permute(featuresTest,[4,3,1,2]);

% 対応するラベルを取得します。
labelsTrain = imdsTrain.Labels;
labelsTest = imdsTest.Labels;

% 5. サポートベクターマシン(SVM)分類器の学習
% 抽出された特徴と対応するラベルを使用して、fitcecoc関数でSVM分類器を学習させます。
% fitcecoc関数はStatistics and Machine Learning Toolbox™が必要です。
fprintf('SVM分類器を学習中...\n');
classifier = fitcecoc(featuresTrain, labelsTrain);
fprintf('SVM分類器の学習が完了しました。\n');

% 6. 分類器の評価
% 学習済みSVM分類器を使用してテストデータで予測を実行します。
YPred = predict(classifier, featuresTest);

% 精度を計算します。精度は、分類器が正しく予測するラベルの割合です。
accuracy = mean(YPred == labelsTest);
fprintf('SVM 分類器のテスト精度: %.2f%%\n', accuracy * 100);

% 混同行列で結果を可視化します。
figure;
confusionchart(labelsTest, YPred, 'Title', 'SVM Classification Confusion Matrix');