% 1. データの準備
rng(42);
num_input_features = 3;
num_samples = 100;
num_classes = 4;
classNames = ["Class1", "Class2", "Class3", "Class4"];

% 入力/教師データの作成
u = rand(num_samples, num_input_features);  % 100 x 3
t_labels = randi([1, num_classes], num_samples, 1);
t = categorical(t_labels, 1:num_classes, classNames);


% 2. ネットワークの作成
% 10個のニューロンを持つ隠れ層が1つあるフィードフォワードネットワークを作成します
% アクティベーションはそれぞれ tanh, linear とします。
hidden_layer_size = 10;

layers = [
    featureInputLayer(num_input_features)
    fullyConnectedLayer(hidden_layer_size)
    tanhLayer
    fullyConnectedLayer(num_classes)
    softmaxLayer
    ];
net = dlnetwork(layers);

% 3. Adamソルバーを使用してネットワークの学習
options = trainingOptions("adam", ...
    MaxEpochs=300, ...
    MiniBatchSize=num_samples, ...
    InitialLearnRate=0.01, ...
    Verbose=false, ...
    Plots="none");

net = trainnet(u, t, net, "crossentropy", options);

% 4. 学習結果の確認
scores = minibatchpredict(net, u);
y_dlnet = scores2label(scores, classNames);
acc_dlnet = mean(y_dlnet == t) * 100;

fprintf('\n=== Classification Results ===\n');
fprintf('Training accuracy:\n');
fprintf('prediction: %.2f%%\n', acc_dlnet);

% 5. 混同行列 (replaces legacy plotconfusion)
figure;
confusionchart(t, y_dlnet);

% 6. 新しいデータに対する予測
new_input = rand(1, num_input_features);
new_scores = minibatchpredict(net, new_input);
[label_dlnet, score_dlnet] = scores2label(new_scores, classNames);

fprintf('\nNew input [%.2f, %.2f, %.2f]:\n', new_input);
fprintf('prediction: %s (score: %.2f)\n', string(label_dlnet), score_dlnet);