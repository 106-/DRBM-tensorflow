#!/usr/bin/env python

import argparse
import datetime
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from DRBM import DRBM
from mltools import LearningLog

parser = argparse.ArgumentParser("DRBM learning script.", add_help=False)
parser.add_argument(
    "learning_config",
    action="store",
    type=str,
    help="path of learning configuration file.",
)
parser.add_argument(
    "learning_epoch", action="store", type=int, help="numbers of epochs."
)
parser.add_argument(
    "-d",
    "--output_directory",
    action="store",
    type=str,
    default="./results/",
    help="directory to output parameter & log",
)
parser.add_argument(
    "-s",
    "--filename_suffix",
    action="store",
    type=str,
    default=None,
    help="filename suffix",
)
parser.add_argument(
    "-p", "--save_parameters", action="store_true", help="save model parameters"
)
args = parser.parse_args()

config = json.load(open(args.learning_config, "r"))
ll = LearningLog(config)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

dtype = config["dtype"]
x_train = x_train.reshape(-1, 3072).astype(dtype) / 255.0
x_test = x_test.reshape(-1, 3072).astype(dtype) / 255.0
y_train = to_categorical(y_train).astype(dtype)
y_test = to_categorical(y_test).astype(dtype)

if "learning_data_limit" in config:
    idx = np.random.choice(np.arange(0, len(x_train)), size=len(x_train), replace=False)
    x_train = x_train[idx[0 : config["learning_data_limit"]]]
    y_train = y_train[idx[0 : config["learning_data_limit"]]]

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(100)

optimizer = tf.keras.optimizers.Adamax(learning_rate=0.002, epsilon=1e-8)
drbm = DRBM(*config["training-layers"], **config["training-args"], dtype=dtype)
drbm.fit_categorical(
    args.learning_epoch,
    len(x_train),
    config["minibatch-size"],
    optimizer,
    train_ds,
    test_ds,
    ll,
)

now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = [
    now,
    "cifar",
    "h" + str(config["training-layers"][1]),
    config["training-args"]["activation"],
]
if args.filename_suffix is not None:
    filename.append(args.filename_suffix)
filename.append("%s.json")
filename = "_".join(filename)

filepath = os.path.join(args.output_directory, filename)
ll.save(filepath % "log")
if args.save_parameters:
    drbm.save(filepath % "model")
