#!/bin/sh

if [ $# -ne 1 ]; then
  echo "requires 1 arguments." 1>&2
  exit 1
fi

DIR="./results/generative/"`date +%Y-%m-%d_%H-%M-%S`"_h200_continuous_sparse/"
mkdir -p $DIR

for i in `seq $1`
do
    ./train_generative.py ./config/generative/h200/continuous_sparse.json 3000 -d $DIR
done