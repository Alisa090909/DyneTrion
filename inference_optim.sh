#!/bin/bash
# Optimized inference script with performance improvements

# source .venv/bin/activate
project_root=./test
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
sample_step=40
n_motion=2
n_frame=16
extrapolation_time=16

test_data=datasets/inference/inference_data.csv
max_protein_num=10
train_step=400000

model_path=./dynamicPDB/DyneTrion/step_400000.pth

noise_scale=1.0
save_root=${project_root}/inference_${TIMESTAMP}/

# Optimization flags (set to empty to use defaults from config file)
# Set to True/False to override configDo
use_amp="True"  # Empty = use config default (True), or set to True/False
use_torch_compile="True"  # Empty = use config default (False), or set to True/False

# Build optional override parameters
override_params=""
if [ -n "$use_amp" ]; then
    override_params="$override_params +use_amp=$use_amp"
fi
if [ -n "$use_torch_compile" ]; then
    override_params="$override_params +use_torch_compile=$use_torch_compile"
fi

echo "========================================"
echo "Running OPTIMIZED inference"
echo "========================================"
echo "Optimizations enabled:"
echo "  - Mixed precision (FP16)"
echo "  - Batch data transfer"
echo "  - GPU tensor caching"
echo "  - Memory optimization"
if [ -n "$override_params" ]; then
    echo "  - Config overrides: $override_params"
fi
echo "========================================"

CUDA_VISIBLE_DEVICES=0 python -m DyneTrion.inference_DyneTrion_optimized \
eval.weights_path=${model_path} \
experiment.use_ddp=False \
eval.extrapolation_time=${extrapolation_time} \
eval.name="test_optimized" \
experiment.batch_size=1 \
experiment.noise_scale=${noise_scale} \
experiment.base_root=${save_root} \
experiment.num_loader_workers=1 \
data.frame_time=${n_frame} \
data.motion_number=${n_motion} \
data.frame_sample_step=${sample_step} \
data.test_csv_path=${test_data} \
data.max_protein_num=${max_protein_num} \
${override_params}

echo "========================================"
echo "Optimized inference completed!"
echo "Results saved to: ${save_root}"
echo "========================================"