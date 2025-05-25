DRBM-tensorflow
===

TensorFlowによる制限ボルツマンマシン分類器(Discriminative-RBM; DRBM a.k.a. Classification RBM; ClassRBM)の実装.

# セットアップ & モジュールインストール
_Python 3.12 くらいが必要_
```
$ git clone https://github.com/106-/DRBM-tensorflow.git
$ cd DRBM-tensorflow
```
サブモジュールのクローン
```
$ git submodule update --init --recursive
```
必要モジュールのインストール
```
$ pip install -r ./requirements.txt
```

# 基本的な使用方法
このプログラムには2つの実行モードがあります：人工データの学習またはカテゴリカルデータの学習。

## 人工データの学習

このモードでは、学習モデルがランダムに生成された生成モデルからサンプリングされた人工データで学習を行います。
これは実用的ではありませんが、生成モデルと学習モデル間の汎化誤差を直接測定するのに有用です。
このプログラムは、生成モデルと学習モデル間の汎化誤差として、エポックごとにカルバック・ライブラー発散（KLD）を計算します。

### 設定ファイル
設定ファイルには以下を記述する必要があります：
- 生成/学習モデルの層（それぞれ入力層、隠れ層、出力層）
- 隠れ層の活性化関数（[下記](#隠れ層の活性化関数)参照）
- dtype（[TensorFlowのデータタイプ](https://www.tensorflow.org/api_docs/python/tf/dtypes/DType)）
- ミニバッチサイズ
- 学習データサイズ

```json
{
    "generative-layers": [20, 50, 10],
    "generative-args": {
        "activation": "continuous"
    },
    "training-layers": [20, 50, 10],
    "training-args": {
        "activation": "continuous"
    },
    "dtype": "float64",
    "minibatch-size": 50,
    "datasize": 500
}
```
このような設定ファイルでプログラムを実行するには：
```
$ ./train_generative.py (設定ファイルへのパス) (学習エポック数)
```
設定ファイルの例は `./config/generative` にあります。

## カテゴリカルデータの学習
このモードでは、学習モデルが分類問題で学習を行います。
このモードは実世界のデータの学習に使用されます。
各エポックで、このプログラムは学習データとテストデータの誤分類率と負の対数尤度を計算します。

`train_mnist.py`、`train_fashion_mnist.py`、`train_olivetti.py`、`train_urban.py` は学習データがモジュールまたはリポジトリに含まれており、直接実行できます。

`train_cifar.py` は学習データが含まれていないため、自分で準備する必要があります。

### 設定ファイル
設定ファイルには以下を記述します：
- 学習モデルの層（それぞれ入力層、隠れ層、出力層）
- 隠れ層の活性化関数（[下記](#隠れ層の活性化関数)参照）
- dtype（[TensorFlowのデータタイプ](https://www.tensorflow.org/api_docs/python/tf/dtypes/DType)）
- ミニバッチサイズ
- 学習データサイズ
- テストデータに追加する白色ノイズの分散（オプション）

人工データ学習の設定ファイルに加えて、テストデータに追加する白色ノイズの分散を設定できます。

```json
{
    "training-layers": [784, 50, 10],
    "training-args": {
        "activation": "continuous"
    },
    "dtype": "float64",
    "minibatch-size": 100,
    "learning_data_limit": 1000,
    "test_noise_std": 250
}
```
このような設定ファイルでプログラムを実行するには：
```
$ ./train_mnist.py (設定ファイルへのパス) (学習エポック数)
```
設定ファイルの例は `./config/mnist` にあります。

# 隠れ層の活性化関数
元の論文[2]では、隠れ層は0または1を取るため、活性化関数はソフトプラス関数となります。
しかし、私の研究では多値隠れノード[3]とスパース正則化を適用しています。
これは設定ファイルの `activation` 設定を変更することで簡単に変更できます。
これらは `hidden_marginalize.py` で定義されています。
以下は隠れ層とその設定の表です。

|        `activation` |          隠れ変数 |                  備考 |
|:-------------------:|:-------------------------:|:---------------------:|
|          `original` |                     {0,1} |       [2]で提案 |
|            `double` |                  {-1, +1} |       [3]で提案 |
|            `triple` |               {-1, 0, +1} |       [3]で提案 |
|        `continuous` |                  [-1, +1] |       [3]で提案 |
|             `esrbm` |      {0, 1} & スパース項 |          [4]で提案 |
|     `triple_sparse` | {-1, 0, +1} & スパース項 |          [5]で提案 |
| `continuous_sparse` |    [-1, +1] & スパース項 |          [5]で提案 |

# 参考文献
- [1]: H. Larochelle and Y. Bengio: [Classification using discriminative restricted boltzmann machines](http://www.dmi.usherb.ca/~larocheh/publications/icml-2008-discriminative-rbm.pdf), Proceedings of the Twenty-fifth International Conference on Machine Learning (ICML'08), pp. 536–543, 2008.
- [2]: H. Larochelle, M. Mandel, R. Pascanu, and Y. Bengio: [Learning algorithms for the classification restricted boltzmann machine](http://www.jmlr.org/papers/volume13/larochelle12a/larochelle12a.pdf), The Journal of Machine Learning Research, Vol. 13, No. 1, pp. 643–669, mar 2012.
- [3]: Y. Yokoyama, T. Katsumata and M. Yasuda: [Restricted Boltzmann Machine with Multivalued Hidden Variables: a model suppressing over-fitting](https://arxiv.org/pdf/1811.12587.pdf), The Review of Socionetwork Strategies, Vol.13, no.2, pp.253-266, 2019.
- [4]: Wei, Jiangshu & Lv, Jiancheng & Yi, Zhang. (2018). A New Sparse Restricted Boltzmann Machine. International Journal of Pattern Recognition and Artificial Intelligence. 33. 10.1142/S0218001419510042. 
- [5]: M. Yasuda and T. Katsumata: [Discriminative restricted Boltzmann machine with trainable sparsity](https://www.jstage.jst.go.jp/article/nolta/14/2/14_207/_pdf), Nonlinear Theory and Its Applications, IEICE, Vol. 14, no. 2, pp. 207–214, 2023.
