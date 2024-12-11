#!/usr/bin/env bash

set -x
NGPUS=$1
PORT=$2
PY_ARGS="${@:3}"

torchrun --nproc_per_node=${NGPUS} --master_port=${PORT} train.py --launcher pytorch ${PY_ARGS}
