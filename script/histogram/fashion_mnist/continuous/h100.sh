#!/bin/sh

if [ $# -ne 1 ]; then
  echo "requires 1 arguments." 1>&2
  exit 1
fi

DIR="./results/histogram_fashion_mnist/"`date +%Y-%m-%d_%H-%M-%S`"_h100_continuous_sparse/"
mkdir -p $DIR

for i in `seq $1`
do
    ./train_fashion_mnist.py ./config/histogram/continuous_mnist/h100.json 100 -d $DIR -p
done