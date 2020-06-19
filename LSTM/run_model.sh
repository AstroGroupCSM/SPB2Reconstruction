#!/bin/bash

#export CUDA_VISIBLE_DEVICES=6
export PIPE_FILE="$PWD/run_model.out"
export CUDA_VISIBLE_DEVICES=6

#if [ "$WHOAMI" == "czh" ]
#  then
#    DATA_DIR="/home/czh/datasets/george_data/v2"
#    BATCH_SIZE=2
#  else
#    DATA_DIR="/home/datasets/george_data/v2"
#    export CUDA_VISIBLE_DEVICES=5
#    BATCH_SIZE=48
#fi

#DATA_DIR="/home/czh/datasets/george_data/v2"
#DATA_DIR="/home/datasets/george_data/v2"
#DATA_DIR="/home/czh/datasets/george_data"
DATA_DIR="/home/datasets/george_data"

#BATCH_SIZE=4
BATCH_SIZE=36

echo "Piping output to $PIPE_FILE..."

cd code

nohup python3 run_model.py --data "$DATA_DIR" \
                           --out "../out" \
                           --train True \
                           --eval True \
                           --epochs 20 \
                           --model_type CNN \
                           --rnn_dim 64 \
                           --batch_size "$BATCH_SIZE" \
                           --only_high_signal True \
                           --balance_data False \
                           --supersample True \
                           --normalize_data True \
                           --weight_xent_loss False \
                           --n_conv_layers 10 \
                           --conv_channels 3 \
                           --schedule_weight_xent_loss False \
                           --schedule_weight_xent_loss_epoch_start 15 \
                           --schedule_weight_xent_loss_epoch_end 19 \
                           --schedule_weight_xent_loss_noise_start_weight 1.0 \
                           --schedule_weight_xent_loss_noise_final_weight 0.9 \
                           --warmup_proportion 0.0 \
                           --do_transforms True \
                           --transform_prob 0.5 \
                           --do_frame_swaps False \
                           --epoch_to_start_frame_swaps 3 \
                           --pct_frames_to_swap 0.15 \
                           --port "19343" > "$PIPE_FILE" &