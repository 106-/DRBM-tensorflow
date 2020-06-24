#!/bin/sh

if [ $# -ne 1 ]; then
  echo "requires 1 arguments." 1>&2
  exit 1
fi

DIR="./results/olivetti/"`date +%Y-%m-%d_%H-%M-%S`"_h300_esrbm/"
mkdir -p $DIR

for i in `seq $1`
do
    ./train_olivetti.py ./config/olivetti/esrbm.json 100 -d $DIR
done