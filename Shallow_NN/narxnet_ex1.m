% Seed 固定 (再現性確保のため)
rng(21)

% 入出力データ作成
% 80サンプル分を学習データとし
% 20サンプル分を予測データとする
[X,T] = simpleseries_dataset;
Xnew = X(81:100); % 未来用データ
Tnew =T(81:100); % 未来用データ
X = X(1:80); % 学習用データ
T = T(1:80); % 学習用データ

% NARX Network の作成
net = narxnet(0:2,1:2,11); % 入力遅延を 0 ～2/ 出力遅延を 1～ 2 サンプルとする
[Xs,Xi,Ai,Ts] = preparets(net,X,{},T); % 時系列ネットワークへ入力するためにデータを準備
net = train(net,Xs,Ts,Xi,Ai); % 学習

% 予測処理の前準備
netc = closeloop(net); % ネットワークの閉ループ化

[Xs,Xi,Ai,Ts] = preparets(netc, X(end-2:end), {}, T(end-2:end) );
%[Xs,Xi,Ai,Ts] = preparets(netc, X(end-2:end), {}, con2seq(zeros(1,3)) ); % Feedback Delay の状態を0

% 予測処理 (test)
y2 = netc(Xnew, Xi, Ai);
y2 = seq2con(y2);
y2 = y2{1};

% 結果確認 およびグラフ化
Tnew = seq2con(Tnew);
Tnew = Tnew{1};
T = seq2con(T);
T = T{1};

figure
plot(1:100,[T,Tnew],'b-o',1:100,[T,y2],'r--*')
legend({'Reference';'Simulation (From 81 - 100)'})
grid on