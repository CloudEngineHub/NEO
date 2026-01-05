# NEO Series: Native Vision-Language Models

This repository provides a training framework for NEO models. The are two steps to use our code:

Customize your dataset: prepare data, implement the config
Modify training scripts:

## Contents
- [Install](#install)
- [Repository Structure](#repository-structure)
- [Custom Dataset Configuration](#custom-dataset-configuration)
- [Usage](#usage)

## Install

#### Environment

```bash
git clone https://github.com/EvolvingLMMs-Lab/NEO.git
cd NEO/VLMTrainKit
conda create -n neo python=3.12 -y
conda activate neo

pip install --upgrade pip
pip install .
```

#### Preparation    

[NOTE]: Download `Qwen3 Model`:
- [Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base)
- [Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base). 


## Repository Structure

### `neo/train/`

- `trainer.py`: Main trainer updated from Huggingface Trainer
- `argument.py`: Dataclasses for model, data and training arguments

### `neo/data/`

- `__init__.py`: Contains datasets configs
- `data_processor.py`: Data processing module for NEO models

## Custom Dataset Configuration

The customized data should have the format like:

### JSON Data Structure

**Media Specification**:

- `image`: Contains path to the media file (required)
- Media tags in prompts:
  - `<image>` for image understanding tasks
- `conversations`: contains the questions and answers

### Example Instances:

**Single Image Example**:
```json
{
  "image": "demo.jpg",
  "width": 335,
  "height": 500,
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nSummarize the content of this picture."
    },
    {
      "from": "gpt",
      "value": "A wooden chair in the living room"
    }
  ]
}
```

**Packed Data Example**:
```json
[
    {
        "image": "images/001.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nWhat's the main object in this picture?"
            },
            {
                "from": "gpt",
                "value": "A red apple on a wooden table"
            }
        ]
    },
    {
        "image": "images/002.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nWhat's the main object in this picture?"
            },
            {
                "from": "gpt",
                "value": "A green orange on a plastic table"
            }
        ]
    }
]
```

### Dataset config for training
### Dataset Definition Structure

1. **Create a dataset dictionary** in the format in the file `data/__init__.py`:
```python
DATASET_NAME = {
    "annotation_path": "/path/to/annotations.json",
    "data_path": "/path/to/image/data",  # Can be empty if paths are in annotations
}
```

2. **Register your dataset** by adding it to the `data_dict`:
```python
data_dict = {
    "your_dataset_name": DATASET_NAME,
    # ... other datasets
}
```

### Sampling Rate Control

You can optionally specify sampling rates by appending `%X` to the dataset name:
- `"dataset_name%50"` will sample 50% of the data
- `"dataset_name%20"` will sample 20% of the data


2. Use it in training:
```python
dataset_names = ["my_dataset%50"]  # Will use 50% of your dataset
configs = data_list(dataset_names)
```

### Notes  
- The `annotation_path` should point to a JSON or JSONL file containing your dataset annotations.  
- The `data_path` can be left empty if the image paths in the annotations are absolute.  
- Sampling rates are applied per-dataset when multiple datasets are specified.  
- The training data should strictly follow this format:  
  - One `<image>` tag in the question must correspond to exactly one image file  
  - Similarly, `<video>` tags must correspond to video files  



## Usage

To train a model:
```bash
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
    --extra_num_layers 12 \   # Number of pre-buffer layers
    --num_hidden_layers 28 \  # Total number of layers in the model
    --train_buffer \  # Whether to train only the prebuffer layers
    --num_train_epochs 1 \
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
```