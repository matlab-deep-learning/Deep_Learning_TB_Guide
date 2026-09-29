### 転移学習の例題 (MATLAB®コード)

転移学習は、事前学習済みのネットワークを新しいタスクの学習の開始点として利用する深層学習の一般的な手法です。この手法は、ゼロからネットワークを学習させるよりもはるかに簡単で時間がかからないことが多く、少ない学習イメージでも新しいタスクに特徴を高速に転移できるという利点があります。

以下に、ResNet-18学習済みネットワークを用いて`MerchData.zip`に含まれる5クラスの画像を分類する転移学習の例題（サンプルMATLABコード）と解説を示します。

**必要なツールボックス**:
*   Deep Learning Toolbox™
*   Deep Learning Toolbox Model for ResNet-18 Network サポートパッケージ

```matlab
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
```

**コードの解説:**

1.  **データセットの準備**:
    *   `MerchData.zip`ファイルを解凍し、その中の画像から`imageDatastore`オブジェクトを作成します。`imageDatastore`は、フォルダ名に基づいて自動的に画像をラベル付けし、メモリに収まらない大きなデータセットも効率的に扱えます。
    *   `splitEachLabel`関数を使って、データセットを学習(`imdsTrain`)、検証(`imdsValidation`)、テスト(`imdsTest`)の3つのセットにランダムに分割します。
    *   クラス名とクラス数も抽出します。

2.  **学習済みネットワークの読み込みと変更**:
    *   `imagePretrainedNetwork("resnet18")`を使用して、事前学習済みのResNet-18ネットワークを読み込みます。ResNet-18は、ImageNetデータセットで学習されており、広範囲の画像に対する豊富な特徴表現を学習済みです。
    *   ネットワークの入力サイズ(`inputSize`)を取得します。これは、入力画像をネットワークに供給する前にリサイズするために必要です。
    *   転移学習では、元のネットワークの最終層を新しいタスク（この場合は5クラス分類）に適応させるために変更します。ここでは、`layerGraph`を使ってネットワークをグラフとして扱い、既存の最終分類層（ResNet-18の場合`prob`層）を削除し、新しい全結合層、ソフトマックス層、分類層を追加して、MerchDataの5つのクラスに対応させます。

3.  **データ拡張と学習オプションの指定**:
    *   `augmentedImageDatastore`を作成し、学習イメージのサイズをネットワークの入力サイズに合わせて自動的に変更するよう設定します。
    *   `imageDataAugmenter`を使用して、ランダムな水平反転や平行移動などのデータ拡張を適用します。これは、ネットワークが過適合するのを防ぎ、汎化性能を高めるのに役立ちます。
    *   `trainingOptions`関数を使って、ネットワークの学習方法に関する設定（ソルバーの種類、初期学習率、エポック数、ミニバッチサイズ、検証データの使用、学習進捗のプロット表示など）を定義します。

4.  **ネットワークの学習**:
    *   `trainnet`関数を使用して、変換された学習データ(`augimdsTrain`)、変更されたネットワーク(`net`)、損失関数(`'crossentropy'`)、および学習オプション(`options`)を指定してネットワークの学習を開始します。これにより、事前学習済みの重みを利用して、新しいMerchDataタスクに効率的に適応します。GPUが利用可能な場合、`trainnet`は自動的にGPUを使用し、学習を高速化します。

5.  **ネットワークのテスト**:
    *   学習が完了したら、`minibatchpredict`関数と`scores2label`関数を使って、テストデータセット(`augimdsTest`)に対するネットワークの予測を行います。`minibatchpredict`もGPUを利用可能です。
    *   予測結果と実際のラベルを比較し、`mean`関数で全体的な分類精度を計算します。
    *   `confusionchart`関数を使用して、予測と真のラベルの間の混同行列を視覚的に表示し、ネットワークがどのクラスを正しく/誤って分類したかを理解できます。

6.  **新しいデータでの予測の実行**:
    *   学習済みネットワーク(`trainedNet`)を新しい単一の画像に適用して、そのクラスを予測する方法を示します。
    *   予測したい画像を読み込み、ネットワークの入力サイズに合わせてリサイズし、`single`データ型に変換します。
    *   `canUseGPU`でGPUの利用可能性を確認し、可能であれば画像をGPUに転送します。
    *   `predict`関数を使って画像の予測スコアを取得し、`scores2label`で最も可能性の高いラベルとそのスコアを抽出します。
    *   最後に、画像と予測結果をタイトルと共に表示します。

このサンプルコードは、MATLAB、 Deep Learning Toolbox を使用した転移学習の一般的なワークフローをカバーしており、提供されたソース情報に基づいて詳細に説明されています。
