#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS="${@:2}"

torchrun --nproc_per_node=${NGPUS} train.py --launcher pytorch ${PY_ARGS}
