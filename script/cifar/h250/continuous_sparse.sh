#!/bin/sh

if [ $# -ne 1 ]; then
  echo "requires 1 arguments." 1>&2
  exit 1
fi

DIR="./results/cifar/"`date +%Y-%m-%d_%H-%M-%S`"_h250_continuous_sparse/"
mkdir -p $DIR

for i in `seq $1`
do
    ./train_cifar.py ./config/cifar/h250/continuous_sparse.json 300 -d $DIR -p
done