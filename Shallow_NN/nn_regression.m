% 1. 学習データの準備
% 例として、ランダムなデータを生成します。
% 実際のアプリケーションでは、ここに実際のデータを読み込みます。
numSamples = 100;  % サンプル数
numInputs = 3;     % 入力数
numOutputs = 2;    % 出力数

% 入力/教師データ
% MATLABでは通常、入力を行列として、行が観測値、列が特徴として扱います。
P = rand(numSamples, numInputs);       % 入力データ (100 サンプル x 3 特徴)
T = rand(numSamples, numOutputs) * 10; % 教師データ (100 サンプル x 2 出力)

% 2. ネットワークの作成
% 10個のニューロンを持つ隠れ層が1つあるフィードフォワードネットワークを作成します。
% アクティベーションはそれぞれ tanh, linear とします。% アクティベーションはそれぞれ tanh, linear とします。
hiddenLayerSize = 10;

layers = [
    featureInputLayer(numInputs)
    fullyConnectedLayer(hiddenLayerSize)
    tanhLayer
    fullyConnectedLayer(numOutputs)
    ];

net = dlnetwork(layers);

% 3. Levenberg-Marquardtソルバーを使用してネットワークの学習
% trainnet関数を使用してネットワークに学習させます。
options = trainingOptions("lm");

net = trainnet(P, T, net, "mse", options);

% 4. 推論（予測）
% 学習済みネットワークを使用して新しい入力データに対する予測を行います。
% 新しいデータは、学習データと同じ形式（新しいサンプル数 x 3特徴）である必要があります。
numNewSamples = 5;
P_new = rand(numNewSamples, numInputs); % 新しい入力データ

% predict()関数で予測を実行します。
Y_pred = predict(net, P_new);

fprintf('新しい入力データに対する予測結果:\n');
disp(Y_pred);