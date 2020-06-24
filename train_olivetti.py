#!/usr/bin/env python

import json
import argparse
import numpy as np
import tensorflow as tf
import datetime
import os
from DRBM import DRBM
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_olivetti_faces
from mltools import LearningLog

parser = argparse.ArgumentParser("DRBM learning script.", add_help=False)
parser.add_argument("learning_config", action="store", type=str, help="path of learning configuration file.")
parser.add_argument("learning_epoch", action="store", type=int, help="numbers of epochs.")
parser.add_argument("-d", "--output_directory", action="store", type=str, default="./results/", help="directory to output parameter & log")
parser.add_argument("-s", "--filename_suffix", action="store", type=str, default=None, help="filename suffix")
args = parser.parse_args()

config = json.load(open(args.learning_config, "r"))
ll = LearningLog(config)

data = fetch_olivetti_faces()
y = data["target"]
x = data["data"]

y = to_categorical(y)

dtype = config["dtype"]
x = x.astype(dtype)
y = y.astype(dtype)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)

if "learning_data_limit" in config:
    idx = np.random.choice(np.arange(0, len(x_train)), size=len(x_train), replace=False)
    x_train = x_train[idx[0:config["learning_data_limit"]]]
    y_train = y_train[idx[0:config["learning_data_limit"]]]

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(100)

optimizer = tf.keras.optimizers.Adamax(learning_rate=0.002, epsilon=1e-8)
drbm = DRBM(*config["training-layers"], **config["training-args"], dtype=dtype)
drbm.fit_categorical(args.learning_epoch, len(x_train), config["minibatch-size"], optimizer, train_ds, test_ds, ll)

now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = [
    now,
    "olivetti",
    "h"+str(config["training-layers"][1]),
    config["training-args"]["activation"]
]
if args.filename_suffix is not None:
    filename.append(args.filename_suffix)
filename.append("%s.json")
filename = "_".join(filename)

filepath = os.path.join(args.output_directory, filename%"log")
ll.save(filepath)
