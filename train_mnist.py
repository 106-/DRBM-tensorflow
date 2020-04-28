
import json
import argparse
import tensorflow as tf
import datetime
import os
from DRBM import DRBM
from tensorflow.keras.utils import to_categorical
from mltools import LearningLog

parser = argparse.ArgumentParser("DRBM learning script.", add_help=False)
parser.add_argument("learning_config", action="store", type=str, help="path of learning configuration file.")
parser.add_argument("learning_epoch", action="store", type=int, help="numbers of epochs.")
parser.add_argument("-d", "--output_directory", action="store", type=str, default="./results/", help="directory to output parameter & log")
parser.add_argument("-s", "--filename_suffix", action="store", type=str, default=None, help="filename suffix")
args = parser.parse_args()

config = json.load(open(args.learning_config, "r"))
ll = LearningLog(config)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = (x_train / 255.0), (x_test / 255.0)

dtype = config["dtype"]

x_train = x_train.reshape(-1, 784).astype(dtype)
x_test = x_test.reshape(-1, 784).astype(dtype)
y_train = to_categorical(y_train).astype(dtype)
y_test = to_categorical(y_test).astype(dtype)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(config["minibatch-size"])
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(100)

optimizer = tf.keras.optimizers.Adamax(learning_rate=0.002)
drbm = DRBM(*config["training-layers"], **config["training-args"], dtype=dtype)
drbm.fit_categorical(args.learning_epoch, optimizer, train_ds, test_ds, ll)

now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = [
    now,
    "mnist",
    "h"+str(config["training-layers"][1]),
    config["training-args"]["activation"]
]
if args.filename_suffix is not None:
    filename.append(args.filename_suffix)
filename.append("%s.json")
filename = "_".join(filename)

filepath = os.path.join(args.output_directory, filename%"log")
ll.save(filepath)