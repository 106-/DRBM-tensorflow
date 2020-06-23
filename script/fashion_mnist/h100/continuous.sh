#!/bin/sh

if [ $# -ne 1 ]; then
  echo "requires 1 arguments." 1>&2
  exit 1
fi

DIR="./results/fashion_mnist/"`date +%Y-%m-%d_%H-%M-%S`"_h100_continuous/"
mkdir -p $DIR

for i in `seq $1`
do
    ./train_fashion_mnist.py ./config/mnist/h100/continuous.json 100 -d $DIR
done