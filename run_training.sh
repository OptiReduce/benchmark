#!/bin/bash

# Check if all required arguments are provided
if [ -z "$5" ]; then
    echo "Usage: $0 <MASTER_ADDR> <RANK> <NODES> <DEV> <MODEL>"
    echo "Available models: vgg19, bert, bart, roberta, gpt2"
    echo "Set RUN_ALL=1 to run all communication schemes"
    exit 1
fi

# Set environment variables
export MASTER_ADDR=$1
export RANK=$2
export NODES=$3
export DEV=$4
export MODEL=$5
export CUBLAS_WORKSPACE_CONFIG=:16:8
export RUN_ALL=${RUN_ALL:-0}  # Default to 0 if not set

# Print the environment variables
echo "Environment variables set:"
echo "CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "RANK=$RANK"
echo "NODES=$NODES"
echo "DEV=$DEV"
echo "MODEL=$MODEL"
echo "RUN_ALL=$RUN_ALL"

# Assign variables based on the model
case $MODEL in
    vgg19)
        BATCH_SIZE=128
        EPOCHS=150
        TR_TIMEOUT=135
        ;;
    bert)
        BATCH_SIZE=16
        EPOCHS=5
        TR_TIMEOUT=350
        ;;
    roberta)
        BATCH_SIZE=16
        EPOCHS=5
        TR_TIMEOUT=370
        ;;
    bart)
        BATCH_SIZE=8
        EPOCHS=6
        TR_TIMEOUT=370
        ;;
    gpt2)
        BATCH_SIZE=8
        EPOCHS=6
        TR_TIMEOUT=370
        ;;
    *)
        echo "Invalid model specified. Available models: vgg19, bert, bart, roberta, gpt2"
        exit 1
        ;;
esac

# Construct the base command
BASE_COMMAND="python examples/train.py --nr $RANK --nodes $NODES --model $MODEL --epochs $EPOCHS --batch_size $BATCH_SIZE --dev $DEV"

echo "Executing commands..."

# Run all communication schemes if RUN_ALL is set
if [ "$RUN_ALL" = "1" ]; then
    echo "Running all communication schemes..."

    # NCCL schemes
    NCCL_IB_DISABLE=1 $BASE_COMMAND --algo ring --comm nccl --uhc
    sleep 5

    NCCL_IB_DISABLE=1 $BASE_COMMAND --algo tree --comm nccl --uhc
    sleep 5

    # Gloo schemes
    $BASE_COMMAND --algo ring --uhc
    sleep 5

    $BASE_COMMAND --algo bcube --uhc
    sleep 5

    $BASE_COMMAND --algo transpose --uhc
    sleep 5
fi

# Always run OptiReduce
echo "Running OptiReduce..."
taskset -c 1-8 $BASE_COMMAND --algo optireduce --tr_timeout $TR_TIMEOUT --tr_threads_offset 11 --uhc

echo "All commands executed."
