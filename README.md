DRBM-tensorflow
===

TensorFlowによる制限ボルツマンマシン分類器(Discriminative-RBM; DRBM a.k.a. Classification RBM; ClassRBM)の実装.

A implementation of discriminative restricted Boltzmann machine(DRBM or Classification RBM; Class RBM) using TensorFlow.

# Setup & Module Installation
_requires Python 3.x_
```
$ git clone https://github.com/106-/DRBM-tensorflow.git
$ cd DRBM-tensorflow
```
Cloning submodules
```
$ git submodule update --init --recursive
```
Installing required modules
```
$ pip install -r ./requirements.txt
```

# Basic Usage
This program has two running mode: learning artificial data or learning categorical data.

## Learning Artificial Data

In this mode, learning model will train artificial data sampled from randomely generated generative model.
This is not practical, but this is useful for measuring generalization error between generative model and training model directly.
This program will calculate Kullback-Leibler Divergence (KLD) as generalization error between generative and learning model per one epoch.

### Configuration File
Configuration file must describe;
- Layers of generative / training model (input, hidden and output layers, respectively.).
- Activation function of hidden layer (See [below](#-Activation-Function-of-Hidden-Layer)).
- dtype ([datatype of Tensorflow](https://www.tensorflow.org/api_docs/python/tf/dtypes/DType)).
- Size of minibatch.
- Size of learning data.

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
With such a configuration file, the program runs as
```
$ ./train_generative.py (path to setting file) (learning epoch)
```
There are some examples of configuration files in `./config/generative`.

## Learning Categorical Data
In this mode, learning model will train classification problem.
This mode is used for training real-world data.
For each epoch, this program will calcurate the misclassification rate and negative log likelihood of training and test data.

`train_mnist.py`, `train_fashion_mnist.py`, `train_olivetti.py` and `train_urban.py` are either the training data is included in module or in repository and you can run directly.

`train_cifar.py` is not include training data, so you need to prepare it yourself.

### Configuration File
Configuration file describes;
- Layers of training model (input, hidden and output layers, respectively.).
- Activation function of hidden layer (See [below](#-Activation-Function-of-Hidden-Layer)).
- dtype ([datatype of Tensorflow](https://www.tensorflow.org/api_docs/python/tf/dtypes/DType)).
- Size of minibatch.
- Size of learning data.
- Variance of white noise to be added to the test data (optional).

In addition to the learning artificial data configuration file, you can set the variance of the white noise to be added to the test data.

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
With such a configuration file, the program runs as
```
$ ./train_mnist.py (path to setting file) (learning epoch)
```
There are some examples of configuration files in `./config/mnist`.

# Activation Function of Hidden Layer
In the original paper[2], the hidden layer takes 0 or 1, so the activation function will be a softplus function.
However, in my research, I have applied multi-valued hidden nodes[3] and sparse regularization.
This can be easily changed by changing the `activation` setting in the configuration file.
And they are defined in `hidden_margianlize.py`.
The following is a table of hidden layers and its settings.

|        `activation` |          Hidden variables |                  Note |
|:-------------------:|:-------------------------:|:---------------------:|
|          `original` |                     {0,1} |       proposed in [2] |
|            `double` |                  {-1, +1} |       proposed in [3] |
|            `triple` |               {-1, 0, +1} |       proposed in [3] |
|        `continuous` |                  [-1, +1] |       proposed in [3] |
|             `esrbm` |      {0, 1} & sparse term |       proposed in [4] |
|     `triple_sparse` | {-1, 0, +1} & sparse term | proposed in [IPSJ 2020](https://www.ipsj.or.jp/event/taikai/82/index.html) |
| `continuous_sparse` |    [-1, +1] & sparse term | proposed in [IPSJ 2020](https://www.ipsj.or.jp/event/taikai/82/index.html) |

# References
- [1]: H. Larochelle and Y. Bengio: [Classification using discriminative restricted boltzmann machines](http://www.dmi.usherb.ca/~larocheh/publications/icml-2008-discriminative-rbm.pdf), Proceedings of the Twenty-fifth International Conference on Machine Learning (ICML’08), pp. 536–543, 2008.
- [2]: H. Larochelle, M. Mandel, R. Pascanu, and Y. Bengio: [Learning algorithms for the classification restricted boltzmann machine](http://www.jmlr.org/papers/volume13/larochelle12a/larochelle12a.pdf), The Journal of Machine Learning Research, Vol. 13, No. 1, pp. 643–669, mar 2012.
- [3]: Y. Yokoyama, T. Katsumata and M. Yasuda: [Restricted Boltzmann Machine with Multivalued Hidden Variables: a model suppressing over-fitting](https://arxiv.org/pdf/1811.12587.pdf), The Review of Socionetwork Strategies, Vol.13, no.2, pp.253-266, 2019.
- [4]: Wei, Jiangshu & Lv, Jiancheng & Yi, Zhang. (2018). A New Sparse Restricted Boltzmann Machine. International Journal of Pattern Recognition and Artificial Intelligence. 33. 10.1142/S0218001419510042. 
