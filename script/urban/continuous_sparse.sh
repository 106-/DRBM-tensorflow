#!/bin/sh

if [ $# -ne 1 ]; then
  echo "requires 1 arguments." 1>&2
  exit 1
fi

DIR="./results/urban/"`date +%Y-%m-%d_%H-%M-%S`"_h300_continuous_sparse/"
mkdir -p $DIR

for i in `seq $1`
do
    ./train_urban.py ./config/urban/continuous_sparse.json 200 -d $DIR
done