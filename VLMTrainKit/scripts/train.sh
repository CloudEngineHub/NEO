#!/bin/bash

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
# For detailed logic, refer to: neo/model/build.py build_model function
mllm=""  # Path to pre-trained NEO model for SFT (Supervised Fine-Tuning) on top of an existing checkpoint
llm=""  # Path to the base LLM model for training NEO from scratch
tokenizer=""  # Path to the tokenizer

# Training hyperparameters
lr=2e-4
batch_size=1
grad_accum_steps=1

# Training entry point
entry_file=neo/train/train.py

# Dataset configuration (replace with public dataset names)
datasets=""

# Output configuration
run_name="neo-baseline"
output_dir=./output

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${mllm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --dtype bfloat16 \
    --output_dir ${output_dir} \
    --extra_num_layers 12 \  # Number of pre-buffer layers
    --num_hidden_layers 28 \  # Total number of layers in the model
    --train_buffer \  # Whether to train only the prebuffer layers
    --num_train_epochs 0.5 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 262144 \
    --min_pixels 12544 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --logging_steps 1 \
    --model_max_length 8096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to tensorboard"

# Set PYTHONPATH to project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Launch training
torchrun --nproc_per_node=2 \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}