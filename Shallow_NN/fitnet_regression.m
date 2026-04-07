% 1. 学習データの準備
% 例として、ランダムなデータを生成します。
% 実際のアプリケーションでは、ここに実際のデータを読み込みます。
numSamples = 100;  % サンプル数
numInputs = 3;     % 入力数
numOutputs = 2;    % 出力数

% 3入力のデータ (例: 3x100の行列, 各列が1つの観測値)
% MATLABの浅層ネットワーク関数は通常、入力を行列として、行が特徴、列が観測値として扱います。
P = rand(numInputs, numSamples); % 入力データ (3特徴 x 100サンプル)

% 2出力のターゲットデータ (例: 2x100の行列)
T = rand(numOutputs, numSamples) * 10; % ターゲットデータ (2出力 x 100サンプル)

% 2. ネットワークの作成
% fitnet関数を使用して、10個のニューロンを持つ隠れ層が1つあるフィードフォワードネットワークを作成します。
% net = fitnet(hiddenSizes)
% hiddenSizesは隠れ層のニューロン数を指定する行ベクトルです。
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);

% ネットワークの表示 (オプション)
% view(net);

% 3. ネットワークの学習
% train関数を使用してネットワークに学習させます。
% fitnetはデフォルトでデータの分割（学習、検証、テスト）を自動的に行います。
% trainnet関数と異なり、浅層ネットワークの学習はtrain関数を使用します。
[net, tr] = train(net, P, T);

% 学習後のパフォーマンスの確認 (オプション)
% plotperform(tr)

% 4. 推論（予測）
% 学習済みネットワークを使用して新しい入力データに対する予測を行います。
% 新しいデータは、学習データと同じ形式（3特徴 x 新しいサンプル数）である必要があります。
numNewSamples = 5;
P_new = rand(numInputs, numNewSamples); % 新しい入力データ

% net()またはsim()関数で予測を実行します。
Y_pred = net(P_new);

fprintf('新しい入力データに対する予測結果:\n');
disp(Y_pred);
