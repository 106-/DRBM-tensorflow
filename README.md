DRBM-tensorflow
===

TensorFlowによる制限ボルツマンマシン分類器(Discriminative-RBM; DRBM a.k.a. Classification RBM; ClassRBM)の実装.

An implementation of discriminative restricted Boltzmann machine (DRBM or Classification RBM; Class RBM) using TensorFlow.

# Setup & Module Installation
_requires Python 3.12 and [uv](https://docs.astral.sh/uv/)_
```
$ git clone https://github.com/106-/DRBM-tensorflow.git
$ cd DRBM-tensorflow
```
Installing required modules
```
$ uv sync
```

To run a script:
```
$ uv run python train_mnist.py (path to setting file) (learning epoch)
```

# Basic Usage
This program has two running modes: learning artificial data or learning categorical data.

## Learning Artificial Data

In this mode, the learning model will train on artificial data sampled from a randomly generated generative model.
This is not practical, but it is useful for directly measuring the generalization error between the generative model and the training model.
This program will calculate the Kullback-Leibler Divergence (KLD) as the generalization error between the generative and learning models per epoch.

### Configuration File
The configuration file must describe:
- Layers of the generative/training model (input, hidden, and output layers, respectively).
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
$ uv run python train_generative.py (path to setting file) (learning epoch)
```
There are some examples of configuration files in `./config/generative`.

## Learning Categorical Data
In this mode, the learning model will train on a classification problem.
This mode is used for training real-world data.
For each epoch, this program will calculate the misclassification rate and negative log-likelihood of the training and test data.

`train_mnist.py`, `train_fashion_mnist.py`, `train_olivetti.py`, and `train_urban.py` include training data either in the module or in the repository, and you can run them directly.

`train_cifar.py` does not include training data, so you need to prepare it yourself.

### Configuration File
The configuration file describes:
- Layers of the training model (input, hidden, and output layers, respectively).
- Activation function of hidden layer (See [below](#-Activation-Function-of-Hidden-Layer)).
- dtype ([datatype of Tensorflow](https://www.tensorflow.org/api_docs/python/tf/dtypes/DType)).
- Size of minibatch.
- Size of learning data.
- Variance of white noise to be added to the test data (optional).

In addition to the artificial data learning configuration file, you can set the variance of white noise to be added to the test data.

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
$ uv run python train_mnist.py (path to setting file) (learning epoch)
```
There are some examples of configuration files in `./config/mnist`.

# Activation Function of Hidden Layer
In the original paper[2], the hidden layer takes 0 or 1, so the activation function will be a softplus function.
However, in my research, I have applied multi-valued hidden nodes[3] and sparse regularization.
This can be easily changed by changing the `activation` setting in the configuration file.
These are defined in `hidden_marginalize.py`.
The following is a table of hidden layers and its settings.

|        `activation` |          Hidden variables |                  Note |
|:-------------------:|:-------------------------:|:---------------------:|
|          `original` |                     {0,1} |       proposed in [2] |
|            `double` |                  {-1, +1} |       proposed in [3] |
|            `triple` |               {-1, 0, +1} |       proposed in [3] |
|        `continuous` |                  [-1, +1] |       proposed in [3] |
|             `esrbm` |      {0, 1} & sparse term |       proposed in [4] |
|     `triple_sparse` | {-1, 0, +1} & sparse term |       proposed in [5] |
| `continuous_sparse` |    [-1, +1] & sparse term |       proposed in [5] |

# References
- [1]: H. Larochelle and Y. Bengio: [Classification using discriminative restricted boltzmann machines](http://www.dmi.usherb.ca/~larocheh/publications/icml-2008-discriminative-rbm.pdf), Proceedings of the Twenty-fifth International Conference on Machine Learning (ICML’08), pp. 536–543, 2008.
- [2]: H. Larochelle, M. Mandel, R. Pascanu, and Y. Bengio: [Learning algorithms for the classification restricted boltzmann machine](http://www.jmlr.org/papers/volume13/larochelle12a/larochelle12a.pdf), The Journal of Machine Learning Research, Vol. 13, No. 1, pp. 643–669, mar 2012.
- [3]: Y. Yokoyama, T. Katsumata and M. Yasuda: [Restricted Boltzmann Machine with Multivalued Hidden Variables: a model suppressing over-fitting](https://arxiv.org/pdf/1811.12587.pdf), The Review of Socionetwork Strategies, Vol.13, no.2, pp.253-266, 2019.
- [4]: Wei, Jiangshu & Lv, Jiancheng & Yi, Zhang. (2018). A New Sparse Restricted Boltzmann Machine. International Journal of Pattern Recognition and Artificial Intelligence. 33. 10.1142/S0218001419510042. 
- [5]: M. Yasuda and T. Katsumata: [Discriminative restricted Boltzmann machine with trainable sparsity](https://www.jstage.jst.go.jp/article/nolta/14/2/14_207/_pdf), Nonlinear Theory and Its Applications, IEICE, Vol. 14, no. 2, pp. 207–214, 2023.
