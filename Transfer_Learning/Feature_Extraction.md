### 特徴抽出の例題 (MATLAB®コード)

この例では、事前学習済みの `resnet18` ネットワークを使用してイメージから特徴を抽出し、抽出されたこれらの特徴に基づいてサポートベクターマシン (SVM) 分類器を学習し、評価する方法を示します。

**必要なツールボックス**:
*   Deep Learning Toolbox™
*   Statistics and Machine Learning Toolbox™
*   Deep Learning Toolbox Model for ResNet-18 Network サポートパッケージ

```matlab
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

```

**コードの解説**:

1.  **データの読み込みと準備**:
    *   まず、例で使用する`MerchData`データセットをダウンロードし、解凍します。これは、MathWorksの商品の小さなイメージデータセットです。
    *   `imageDatastore`は、イメージファイルを管理し、フォルダ名に基づいて自動的にカテゴリカルラベルを割り当てます。
    *   `splitEachLabel`関数を使用して、データセットを学習用とテスト用にランダムに分割します。この例では、学習に70%、テストに30%を使用しています。

2.  **事前学習済みResNet-18ネットワークの読み込み**:
    *   `imagePretrainedNetwork`関数を呼び出して、事前学習済みの`ResNet-18`ネットワークをワークスペースに読み込みます。このネットワークは、ImageNetデータセットで学習されており、一般的なイメージ特徴を抽出するのに非常に効果的です。
    *   ネットワークの入力サイズ（`inputSize`）を取得し、後続のデータ前処理に利用します。

3.  **特徴抽出のためのデータの準備**:
    *   `augmentedImageDatastore`は、`resnet18`の入力層が要求するサイズ（通常は224x224）に合わせて、入力イメージのサイズを自動的に調整します。これにより、抽出プロセス全体でイメージサイズの一貫性が保たれます。

4.  **特徴の抽出**:
    *   `minibatchpredict`関数は、指定された層から特徴（活性化）を抽出するために使用されます。この例では、`'fc1000'`層から特徴を抽出しています。この層は、`ResNet-18`の1000種類の画像ラベルの特徴を捉えるのに適しています。
    *   `minibatchpredict`関数は、によって得られる featureTrain, featureTest 変数は [サンプル数x特徴量数]の行列として返されます。

5.  **サポートベクターマシン(SVM)分類器の学習**:
    *   抽出された特徴(`featuresTrain`)と、それに対応する学習ラベル(`labelsTrain`)を使用して、`fitcecoc`関数（Statistics and Machine Learning Toolbox™が必要）でSVM分類器を学習させます。SVMは、与えられた特徴空間において最適な分離境界を見つけることで分類を行います。

6.  **分類器の評価**:
    *   学習済みのSVM分類器を使用して、テストデータセットから抽出された特徴(`featuresTest`)のラベルを予測します。
    *   予測されたラベルと実際のテストラベルを比較し、`mean`関数を使用して分類精度を計算します。
    *   `confusionchart`を使用して、予測結果を視覚的に混同行列で表示し、分類器のパフォーマンスを詳細に分析できるようにします。

このワークフローは、深層学習モデルの複雑さを直接扱うことなく、その強力な特徴抽出能力を活用して、特定の分類タスクを効率的に解決するための一般的な手法です。
