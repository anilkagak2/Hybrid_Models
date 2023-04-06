#!/bin/bash
NUM_PROC=$1
shift
torchrun --nproc-per-node=$NUM_PROC hybrid_train.py "$@"

#torchrun --nproc-per-node=$NUM_PROC train.py "$@"
