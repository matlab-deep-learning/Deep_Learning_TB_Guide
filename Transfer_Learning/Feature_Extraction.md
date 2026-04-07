### 特徴抽出の例題 (MATLAB®コード)

この例では、事前学習済みの `resnet18` ネットワークを使用してイメージから特徴を抽出し、抽出されたこれらの特徴に基づいてサポートベクターマシン (SVM) 分類器を学習し、評価する方法を示します。

**必要なツールボックス**:
*   Deep Learning Toolbox™
*   Statistics and Machine Learning Toolbox™
*   Deep Learning Toolbox Model for ResNet-18 Network サポートパッケージ

```matlab
% 1. データの読み込みと準備
% この例では、MathWorks™ Merch データセットを使用します。

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
```

**コードの解説**:

1.  **データの読み込みと準備**:
    *   まず、例で使用する`MerchData`データセットをダウンロードし、解凍します。これは、MathWorksの商品の小さなイメージデータセットです。
    *   `imageDatastore`は、イメージファイルを管理し、フォルダ名に基づいて自動的にカテゴリカルラベルを割り当てます。
    *   `splitEachLabel`関数を使用して、データセットを学習用とテスト用にランダムに分割します。この例では、学習に70%、テストに30%を使用しています。

2.  **事前学習済みResNet-18ネットワークの読み込み**:
    *   `resnet18`関数を呼び出して、事前学習済みの`ResNet-18`ネットワークをワークスペースに読み込みます。このネットワークは、ImageNetデータセットで学習されており、一般的なイメージ特徴を抽出するのに非常に効果的です。
    *   ネットワークの入力サイズ（`inputSize`）を取得し、後続のデータ前処理に利用します。

3.  **特徴抽出のためのデータの準備**:
    *   `augmentedImageDatastore`は、`resnet18`の入力層が要求するサイズ（通常は224x224）に合わせて、入力イメージのサイズを自動的に調整します。これにより、抽出プロセス全体でイメージサイズの一貫性が保たれます。

4.  **特徴の抽出**:
    *   `activations`関数は、指定された層から特徴（活性化）を抽出するために使用されます。この例では、`'fc1000'`層から特徴を抽出しています。この層は、`ResNet-18`の1000種類の画像ラベルの特徴を捉えるのに適しています。
    *   `activations`関数は、入力イメージの各観測値に対して、高さ、幅、チャネル、バッチ（観測値の数）の4次元配列として特徴を返します。`fitcecoc`は、この形式の画像特徴量を直接処理できます。後段の fitcecoc 関数の入力用に、permute 関数を使って、featureTrain, featureTest 変数を [サンプル数x特徴量数]の行列に整形しています。

5.  **サポートベクターマシン(SVM)分類器の学習**:
    *   抽出された特徴(`featuresTrain`)と、それに対応する学習ラベル(`labelsTrain`)を使用して、`fitcecoc`関数（Statistics and Machine Learning Toolbox™が必要）でSVM分類器を学習させます。SVMは、与えられた特徴空間において最適な分離境界を見つけることで分類を行います。

6.  **分類器の評価**:
    *   学習済みのSVM分類器を使用して、テストデータセットから抽出された特徴(`featuresTest`)のラベルを予測します。
    *   予測されたラベルと実際のテストラベルを比較し、`mean`関数を使用して分類精度を計算します。
    *   `confusionchart`を使用して、予測結果を視覚的に混同行列で表示し、分類器のパフォーマンスを詳細に分析できるようにします。

このワークフローは、深層学習モデルの複雑さを直接扱うことなく、その強力な特徴抽出能力を活用して、特定の分類タスクを効率的に解決するための一般的な手法です。
