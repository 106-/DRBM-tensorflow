#!/bin/sh

if [ $# -ne 1 ]; then
  echo "requires 1 arguments." 1>&2
  exit 1
fi

DIR="./results/histogram/"`date +%Y-%m-%d_%H-%M-%S`"_h1000_continuous_sparse/"
mkdir -p $DIR

for i in `seq $1`
do
    ./train_generative.py ./config/histogram/continuous/h1000.json 3000 -d $DIR
done