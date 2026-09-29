%% 1. データ準備
rng(21)

% 入出力データ作成
% 80サンプル分を学習データとし
% 20サンプル分を予測データとする
[X,T] = simpleseries_dataset;
XData = cell2mat(X); % 1 x 100
TData = cell2mat(T); % 1 x 100

XTrain = XData(1:80); % 学習用入力データ
TTrain = TData(1:80); % 学習用教師データ
XTest = XData(81:100); % 予測用の入力データ
TTest = TData(81:100); % 予測用の正解出力データ

%% 2. 非線形自己回帰ニューラルネットワーク(nlarx)
data_train = iddata(TTrain', XTrain', 1);

sys = nlarx(data_train, [2 3 0], idSigmoidNetwork(11));

% Closed-loop prediction using forecast
future_input = iddata([], XTest', 1);
yForecast = forecast(sys, data_train, 20, future_input);
y_nlarx = yForecast.OutputData';

mse_nlarx = mean((TTest - y_nlarx).^2);
fprintf('=== NARX Time Series Results ===\n');
fprintf('nlarx prediction MSE: %.6f\n', mse_nlarx);

%% 3. 長・短期記憶(LSTM)ネットワーク

layers = [
sequenceInputLayer(2)          % [exogenous input; feedback]
lstmLayer(11)                  % Match original 11 hidden neurons
fullyConnectedLayer(1)
];
net = dlnetwork(layers);

% Prepare training data: T x C format (time steps x channels)
feedback_train = [0, TTrain(1:end-1)];
XTrainSeq = [XTrain; feedback_train]';  % 80 x 2

options = trainingOptions("adam", ...
    MaxEpochs=500, ...
    MiniBatchSize=1, ...
    InitialLearnRate=0.005, ...
    GradientThreshold=1, ...
    Verbose=false, ...
    Plots="none");

net = trainnet(XTrainSeq, TTrain', net, "mse", options);

% Closed-loop prediction: warm up state then predict iteratively
net = resetState(net);
XWarmup = [XTrain; feedback_train]';
[~, state] = predict(net, XWarmup);
net.State = state;

y_lstm = zeros(1, 20);
prevTarget = TTrain(end);
for i = 1:20
    inputStep = [XTest(i); prevTarget]';
    [YPred, state] = predict(net, inputStep);
    net.State = state;
    y_lstm(i) = YPred;
    prevTarget = y_lstm(i);
end

mse_lstm = mean((TTest - y_lstm).^2);
fprintf('LSTM prediction MSE:  %.6f\n', mse_lstm);

%% 4. 結果比較
fprintf('\n--- Comparison ---\n');
fprintf('nlarx MSE: %.6f (explicit NARX structure, numerically closest to legacy)\n', mse_nlarx);
fprintf('LSTM MSE:  %.6f (learned dynamics, different architecture)\n', mse_lstm);

figure;
tiledlayout(2,1);
nexttile;
plot(1:100, [TTrain, TTest], 'b-o', 1:100, [TTrain, y_nlarx], 'r--*');
legend('Reference', 'nlarx (SysID)');
title('nlarx: Preserves NARX delay structure');
grid on;

nexttile;
plot(1:100, [TTrain, TTest], 'b-o', 1:100, [TTrain, y_lstm], 'r--*');
legend('Reference', 'LSTM (DL Toolbox)');
title('LSTM: Learns temporal dynamics implicitly');
grid on;

