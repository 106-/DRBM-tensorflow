#!/bin/sh

if [ $# -ne 1 ]; then
  echo "requires 1 arguments." 1>&2
  exit 1
fi

DIR="./results/histogram_mnist/"`date +%Y-%m-%d_%H-%M-%S`"_h100_triple_sparse/"
mkdir -p $DIR

for i in `seq $1`
do
    ./train_mnist.py ./config/histogram/triple_mnist/h100.json 3000 -d $DIR -p
done