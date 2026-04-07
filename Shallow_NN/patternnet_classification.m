% 1. データの準備
% 入力データ (u): 3ユニット（特徴量）x サンプル数
% 例えば、100個のサンプルをランダムに生成します。
num_input_features = 3;
num_samples = 100;
u = rand(num_input_features, num_samples); % 3x100のランダム入力データ

% ターゲットデータ (t): 4クラス分類のため、4xサンプル数 のワンホットエンコーディング形式
% 各サンプルがどのクラスに属するかを示すラベルを生成し、それをワンホットエンコーディングに変換します。
num_classes = 4;
% 各サンプルがランダムに1から4のいずれかのクラスに属するとします。
% 例: [1 2 3 4 1 2 ...]
t_labels = randi([1, num_classes], 1, num_samples);

% クラスラベルをワンホットエンコーディングに変換します。
% ind2vec関数は、インデックスベクトルをワンホット行列に変換します。
t = full(ind2vec(t_labels, num_classes)); % 4x100のワンホットエンコードされたターゲットデータ

% 2. ニューラルネットワークの作成
% patternnet関数を使用して、指定された隠れ層のユニット数（10ユニット）を持つネットワークを作成します。
% patternnetは分類モデルに特化しており、出力層は自動的にターゲットデータの次元（この場合4クラス）と
% 分類に適した活性化関数（通常はSoftmax）に設定されます。
hidden_layer_size = 10;
net = patternnet(hidden_layer_size);

% 3. ネットワークの学習
% train関数を使用して、準備した入力データ (u) とターゲットデータ (t) でネットワークを学習させます。
% 学習プロセス中、データはデフォルトで学習 (70%)、検証 (15%)、テスト (15%) に分割されます。
disp('ネットワークの学習を開始します...');
net = train(net, u, t);
disp('ネットワークの学習が完了しました。');

% 4. 学習結果の確認 (オプション)
% 学習済みネットワークを使って、入力データに対する予測を行います。
y_predicted = net(u);

% 混同行列をプロットして分類性能を確認します。
% plotconfusion(t_original, y_predicted_outputs)
% t はワンホットエンコーディングなので、予測出力もワンホットエンコーディングと見なせる形式になります。
% 通常、plotconfusionはターゲットと出力が同じ形式（ワンホットまたはクラスインデックス）を期待します。
figure;
plotconfusion(t, y_predicted);
title('学習データの混同行列');

% 新しいデータに対する予測の例
disp('新しい入力データに対する予測:');
new_input = rand(num_input_features, 1); % 新しい1サンプルデータ (3ユニット)
new_output_onehot = net(new_input);

% ワンホットエンコーディングから予測クラスを抽出します。
[~, predicted_class_index] = max(new_output_onehot);
fprintf('新しい入力 [%.2f, %.2f, %.2f] に対する予測クラス: %d\n', ...
        new_input(1), new_input(2), new_input(3), predicted_class_index);

% ネットワークの構造と重み、バイアスを確認することも可能です。
% disp('ネットワークの重み (入力層-隠れ層):');
% disp(net.IW{1});
% disp('ネットワークのバイアス (隠れ層):');

% disp(net.b{1});
