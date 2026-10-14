#!/bin/bash
# Usage:
#   sh testing_scripts/batch_demo.sh <input_dir> <output_dir>
#
# Input folder convention (two formats are both supported):
#   Format A: {name}.png + {name}.npy
#   Format B: {name}_rgb.png + {name}_sparse_depth.npy
#
# Output per sample:
#   {name}.npy         -- dense metric depth (float32, metres)
#   {name}.png         -- colorized depth (jet colormap)
#   {name}_16bit.png   -- 16-bit metric depth (1 pixel = 0.001 m)

INPUT_DIR=$1
OUTPUT_DIR=$2

if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: sh testing_scripts/batch_demo.sh <input_dir> <output_dir>"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# model configs
load_dav2=1
resolution=3
pred_confidence_input=1
multi_resolution_learnable_gradients_weights="uniform"
optim_layer_input_clamp=1.0
depth_activation_format='exp'
max_depth=300.0
whiten_sparse_depths=1
backbone='rgbd'

# checkpoints
ckpt=../checkpoints/modelv1.1_best_72epochs.pt

count=0
for rgb_file in "$INPUT_DIR"/*.png; do
    [ -f "$rgb_file" ] || continue
    base=$(basename "$rgb_file" .png)

    # Format B: {name}_rgb.png + {name}_sparse_depth.npy
    if [[ "$base" == *_rgb ]]; then
        name="${base%_rgb}"
        depth_file="$INPUT_DIR/${name}_sparse_depth.npy"
    # Format A: {name}.png + {name}.npy
    else
        name="$base"
        depth_file="$INPUT_DIR/${name}.npy"
    fi

    if [ ! -f "$depth_file" ]; then
        echo "[SKIP] No matching sparse depth for: $name"
        continue
    fi

    echo "[$(date +%H:%M:%S)] Processing: $name"

    python demo.py \
        --max_depth $max_depth --data_normalize_median 1 \
        --num_resolution $resolution --multi_resolution_learnable_gradients_weights $multi_resolution_learnable_gradients_weights \
        --load_dav2 $load_dav2 \
        --gpus 0 \
        --GRU_iters 1 --optim_layer_input_clamp $optim_layer_input_clamp --depth_activation_format $depth_activation_format \
        --whiten_sparse_depths $whiten_sparse_depths --gru_internal_whiten_method median \
        --backbone_mode $backbone --pred_confidence_input $pred_confidence_input \
        --pretrain $ckpt \
        --input_rgb "$rgb_file" \
        --input_depth "$depth_file" \
        --output_dir "$OUTPUT_DIR" \
        --output_name "$name"

    count=$((count + 1))
done

echo ""
echo "Done. Processed $count sample(s). Results saved to: $OUTPUT_DIR"
