#!/bin/sh

if [ $# -ne 1 ]; then
  echo "requires 1 arguments." 1>&2
  exit 1
fi

DIR="./results/histogram_mnist/"`date +%Y-%m-%d_%H-%M-%S`"_h2000_triple_sparse/"
mkdir -p $DIR

for i in `seq $1`
do
    ./train_mnist.py ./config/histogram/triple_mnist/h2000.json 100 -d $DIR -p
done