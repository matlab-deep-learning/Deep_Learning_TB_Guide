Deep Learning Toolbox™ は多岐にわたるアプリケーションをサポートしており、特に近年は時系列、コンピューター ビジョン、自然言語処理、Simulink® との連携など、実用的なワークフローが充実しています。

実用に即した例題を以下のカテゴリに分けて提示いたします。  
これらの例題は、前章までで説明した、転移学習や特徴抽出、事前学習済ネットワーク、データストアなどを組み合わせて作成されています。  

### アプリケーション例の候補 (Deep Learning Toolbox)

#### 1. コンピューター ビジョン (画像処理とオブジェクト検出)

最も一般的な深層学習の用途であり、事前学習済みネットワーク（転移学習）の活用例が豊富です。

| 分野 | 具体的な例題 | 関連技術/ネットワーク | 参照情報 |
| :--- | :--- | :--- | :--- |
| **画像分類 (転移学習)** | **新しいイメージを分類するためのニューラル ネットワークの再学習**。事前学習済みのネットワーク（AlexNet、GoogLeNet、ResNet-18 など）を取得し、少ない学習イメージで新しいタスクに迅速に適用する方法。 | 転移学習 (Transfer Learning)、事前学習済みネットワーク (GoogLeNet, AlexNet, SqueezeNetなど) | [Click](https://jp.mathworks.com/help/deeplearning/ug/retrain-neural-network-to-classify-new-images.html)|
| **画像分類 (特徴抽出)** | **事前学習済みのネットワークを使用したイメージの特徴の抽出**。事前学習済みのネットワーク（ResNet-18）を取得し、少ない学習イメージで新しいタスクに迅速に適用する方法。 | 特徴抽出、事前学習済みネットワーク (ResNet-18)、サポートベクターマシン | [Click](https://jp.mathworks.com/help/deeplearning/ug/extract-image-features-using-pretrained-network.html)|
| **画像分類 (スクラッチ)** | **分類用のシンプルな深層学習ニューラル ネットワークの作成**。事前学習済みのネットワークを使用せず、手書き数字を分類するネットワークを一から作成して学習する方法。 | スクラッチから学習、CNN | [Click](https://jp.mathworks.com/help/deeplearning/ug/create-simple-deep-learning-network-for-classification.html)|
| **画像回帰** | **回帰用の畳み込みニューラル ネットワークの学習**。手書き数字の回転角度などの**連続的な数値**を、イメージから直接予測する。 | CNN、平均二乗誤差損失 (MSE) | [Click](https://jp.mathworks.com/help/deeplearning/ug/train-a-convolutional-neural-network-for-regression.html) |
| **リアルタイム分類** | **深層学習を使用した Web カメラ イメージの分類**。Web カメラからのイメージをリアルタイムで取得し、事前学習済みネットワーク (GoogLeNet) を使用してオブジェクトを分類する例。 | CNN (GoogLeNet)、リアルタイム推論 | [Click](https://jp.mathworks.com/help/deeplearning/ug/classify-images-from-webcam-using-deep-learning.html)|
| **オブジェクト検出** | **YOLO v2/v3/v4/X 深層学習を使用した車両検出や欠陥検出**。イメージ内で特定のオブジェクト（車両、交通標識、プリント基板の欠陥など）の位置とクラスを検出する。| YOLO (You Only Look Once) v2/v3/v4/X、SSD (Single Shot Multibox Detector) | [YOLO v2](https://jp.mathworks.com/help/deeplearning/ug/object-detection-using-yolo-v2.html)  [YOLO v3](https://jp.mathworks.com/help/vision/ug/object-detection-using-yolo-v3-deep-learning.html)   [YOLO v4](https://jp.mathworks.com/help/vision/ug/object-detection-using-yolov4-deep-learning.html)  [YOLO X](https://jp.mathworks.com/help/vision/ug/detect-pcb-defects-using-yolox-deep-learning.html)   [SSD](https://jp.mathworks.com/help/deeplearning/ug/object-detection-using-ssd-deep-learning.html)|
| **セマンティック セグメンテーション** | **路上イメージのピクセル単位のセグメンテーション**。自動運転の応用例として、イメージ内のすべてのピクセルを道路、建物、歩行者などのカテゴリに分類する。| DeepLab v3+、U-Net、膨張畳み込み | [Click](https://jp.mathworks.com/help/deeplearning/ug/semantic-segmentation-using-deep-learning.html)|


#### 2. 時系列とシーケンス処理

LSTM/BiLSTM ネットワークは、時系列データやセンサー データの予測および分類に幅広く使用されています。

| 分野 | 具体的な例題 | 関連技術/ネットワーク | 参照情報 |
| :--- | :--- | :--- | :--- |
| **時系列予測** | **深層学習を使用した時系列予測**。過去の時系列データを使用して、将来の値を予測する。特に LSTM ネットワークによる予測。 | LSTM ネットワーク、複数タイムステップ予測 | [Click](https://jp.mathworks.com/help/deeplearning/ug/time-series-forecasting-using-deep-learning.html) |
| **シーケンス分類** | **深層学習を使用したシーケンスの分類**。正弦波、方形波、三角波、ノコギリ波の 4 つのクラスの波形を分類。 | LSTM/BiLSTM ネットワーク、sequence-to-label 分類 | [Click](https://jp.mathworks.com/help/deeplearning/ug/classify-sequence-data-using-lstm-networks.html) |
| **音声/オーディオ分類** | **CNN-LSTM ネットワークを使用したシーケンス分類**。音声テキストから感情を認識するなど、音声データをスペクトログラムに変換し、CNN-LSTM ネットワークで分類する。| CNN-LSTM ネットワーク、音声コマンド認識 | [Click](https://jp.mathworks.com/help/deeplearning/ug/sequence-classification-using-cnn-lstm-network.html)|

#### 3. テキスト分析 (自然言語処理)

テキスト データの分類や生成に再帰型ネットワーク（LSTM、BiLSTM）や Transformer ベースのモデルが使用されます。

| 分野 | 具体的な例題 | 関連技術/ネットワーク | 参照情報 |
| :--- | :--- | :--- | :--- |
| **テキスト分類** | **深層学習を使用したテキスト データの分類**。LSTM ネットワークを使用して、テキスト（例: レポート）を事象タイプなどのカテゴリに分類する。 | LSTM ネットワーク、畳み込みニューラル ネットワーク (CNN)、単語埋め込み層 (`wordEmbeddingLayer`) | [Click](https://jp.mathworks.com/help/textanalytics/ug/classify-text-data-using-deep-learning.html)|
| **複数ラベル分類** | **深層学習を使用した複数ラベルをもつテキストの分類**。1つのドキュメントに複数の独立したラベル（タグ）を分類する。損失関数としてバイナリ クロスエントロピーを使用する。 | BiLSTM ネットワーク、カスタム損失関数 | [Click](https://jp.mathworks.com/help/deeplearning/ug/multilabel-text-classification-using-deep-learning.html)|
| **テキスト生成** | **深層学習を使用した単語単位のテキスト生成**。LSTM ネットワークに学習させ、単語のシーケンスから次の単語を予測することで、新しいテキストを生成する。| Sequence-to-sequence LSTM ネットワーク、自己符号化器 | [Click](https://jp.mathworks.com/help/textanalytics/ug/word-by-word-text-generation-using-deep-learning.html)|

#### 4. Simulink および組み込みシステムとの連携

学習済みのネットワークを Simulink モデルに組み込み、シミュレーションやコード生成を行う例は実用性が高いです。

| 分野 | 具体的な例題 | 関連技術/ネットワーク | 参照情報 |
| :--- | :--- | :--- | :--- |
| **組み込み予測/分類** | **Simulink でのネットワークの状態の予測と更新**。Simulink の `Stateful Predict` ブロックや `Stateful Classify` ブロック を使用して、再帰型ネットワーク (LSTM) の状態をシミュレーション ステップごとに更新しながら予測を行う。| `Stateful Predict` ブロック、LSTM ネットワーク | [Click](https://jp.mathworks.com/help/deeplearning/ug/classify-update-network-simulink.html)|
| **予知保全 (RUL推定)** | **深層学習ネットワークを使用した Simulink での時系列の予測**。ターボファン エンジンのセンサー データから、エンジンの**残存耐用年数 (RUL)** をサイクル単位で予測する。1. ML上で学習し 2.SL上で予測する2部構成 | LSTM ネットワーク、sequence-to-sequence 回帰、Stateful Predict ブロック | [Click *1](https://jp.mathworks.com/help/deeplearning/ug/sequence-to-sequence-regression-using-deep-learning.html)   [Click *2](https://jp.mathworks.com/help/deeplearning/ug/time-series-prediction-in-simulink-using-deep-learning-network.html)|
| **物理システムのモデリング** | **Simulink における LSTM ネットワークを使用した物理システムのモデリング**。LSTM ネットワークを**バーチャル センサー**として機能する低次元化モデル (ROM) として使用し、物理システムの計算負荷を軽減する。| LSTM-ROM、`Stateful Predict` ブロック | [Click](https://jp.mathworks.com/help/deeplearning/ug/physical-system-modeling-using-lstm-network.html)|
| **バッテリー充電状態推定 (SOC)** | **深層学習を使用したバッテリー充電状態の推定**。バッテリーの SOC (State of Charge) を推定するエンドツーエンドのワークフロー（要件定義、データ準備、学習、圧縮、Simulink への組み込み、コード生成）。| 一般的な深層学習ネットワーク、Simulink の `Predict` ブロック | [Click](https://jp.mathworks.com/help/deeplearning/ug/battery-state-of-charge-estimation-using-deep-learning.html)|

