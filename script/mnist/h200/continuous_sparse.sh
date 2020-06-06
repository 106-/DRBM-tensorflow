#!/bin/sh

if [ $# -ne 1 ]; then
  echo "requires 1 arguments." 1>&2
  exit 1
fi

DIR="./results/mnist/"`date +%Y-%m-%d_%H-%M-%S`"_h200_continuous_sparse/"
mkdir -p $DIR

for i in `seq $1`
do
    ./train_mnist.py ./config/mnist/h200/continuous_sparse.json 100 -d $DIR
done