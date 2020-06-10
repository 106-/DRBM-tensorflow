#!/bin/sh

if [ $# -ne 1 ]; then
  echo "requires 1 arguments." 1>&2
  exit 1
fi

DIR="./results/mnist/"`date +%Y-%m-%d_%H-%M-%S`"_h100_triple/"
mkdir -p $DIR

for i in `seq $1`
do
    ./train_mnist.py ./config/mnist/h100/triple.json 100 -d $DIR
done