#!/bin/bash
# CRATE Training Pipeline
# Runs base training in 50k chunks, with mid+SFT after each checkpoint
#
# Usage: ./train_pipeline.sh [start_step] [end_step] [model_folder]
# Example: ./train_pipeline.sh 0 300000 crate-run-1
#          ./train_pipeline.sh 100000 300000 crate-run-1
#
# Args:
#   start_step: Starting step (use 0 to start fresh without loading)
#   end_step: Ending step
#   model_folder: Unique folder name in .cache/nanochat/ (auto-generated if not provided)

set -e

# Wandb configuration
# To disable wandb: export WANDB_MODE=offline
# To enable wandb: comment out the line below or set WANDB_MODE=online
# export WANDB_MODE=offline
export WANDB_SILENT=false

# Configuration
START_STEP=${1:-100000}
END_STEP=${2:-300000}
# Generate unique model folder if not provided (using timestamp)
if [ -z "$3" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    MODEL_FOLDER="crate-${TIMESTAMP}"
    echo "No model folder specified, using: $MODEL_FOLDER"
else
    MODEL_FOLDER="$3"
fi
CHUNK_SIZE=50000

# Training hyperparameters (adjust for your GPU)
DEPTH=12
ASPECT_RATIO=64
MAX_SEQ_LEN=1024
DEVICE_BATCH_SIZE=24
TOTAL_BATCH_SIZE=49152  # Must be divisible by (24 × 1024 = 24,576)

# Mid/SFT hyperparameters
MID_DEVICE_BATCH=24
MID_SEQ_LEN=1024
MID_TOTAL_BATCH=24576

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

cd /home/robel/nanoCRATE/nanochat

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}CRATE Training Pipeline${NC}"
echo -e "${BLUE}Start: ${START_STEP} → End: ${END_STEP}${NC}"
echo -e "${BLUE}Chunk size: ${CHUNK_SIZE}${NC}"
echo -e "${BLUE}Model folder: ${MODEL_FOLDER}${NC}"
echo -e "${BLUE}Model path: ~/.cache/nanochat/${MODEL_FOLDER}/${NC}"
echo -e "${BLUE}========================================${NC}"

# Create model folder if starting fresh
if [ $START_STEP -eq 0 ]; then
    echo -e "${GREEN}Starting fresh training (step 0) - no model will be loaded${NC}"
    mkdir -p ~/.cache/nanochat/${MODEL_FOLDER}
fi

CURRENT_STEP=$START_STEP

while [ $CURRENT_STEP -lt $END_STEP ]; do
    NEXT_STEP=$((CURRENT_STEP + CHUNK_SIZE))
    
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Phase: Base Training ${CURRENT_STEP} → ${NEXT_STEP}${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    # Base training for this chunk
    # Build command conditionally based on whether we're resuming
    BASE_TRAIN_CMD="python -m scripts.base_train \
        --depth=$DEPTH \
        --aspect_ratio=$ASPECT_RATIO \
        --max_seq_len=$MAX_SEQ_LEN \
        --device_batch_size=$DEVICE_BATCH_SIZE \
        --total_batch_size=$TOTAL_BATCH_SIZE \
        --num_iterations=$NEXT_STEP \
        --eval_every=5000 \
        --sample_every=10000 \
        --save_every=$CHUNK_SIZE \
        --core_metric_every=-1 \
        --model_tag=$MODEL_FOLDER \
        --run=${MODEL_FOLDER}-${NEXT_STEP}"
    
    # Only add --resume_from_step if we're not starting fresh
    if [ $CURRENT_STEP -gt 0 ]; then
        BASE_TRAIN_CMD="$BASE_TRAIN_CMD --resume_from_step=$CURRENT_STEP"
        echo -e "${GREEN}Resuming from step $CURRENT_STEP${NC}"
    else
        echo -e "${GREEN}Starting fresh training (no resume)${NC}"
    fi
    
    # Execute the command
    eval $BASE_TRAIN_CMD
    
    echo ""
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Phase: Midtraining (after ${NEXT_STEP} steps)${NC}"
    echo -e "${YELLOW}========================================${NC}"
    
    # Midtraining
    python -m scripts.mid_train \
        --device_batch_size=$MID_DEVICE_BATCH \
        --max_seq_len=$MID_SEQ_LEN \
        --total_batch_size=$MID_TOTAL_BATCH \
        --model_tag=$MODEL_FOLDER \
        --run=${MODEL_FOLDER}-mid-${NEXT_STEP}
    
    echo ""
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Phase: SFT (after ${NEXT_STEP} steps)${NC}"
    echo -e "${YELLOW}========================================${NC}"
    
    # SFT
    python -m scripts.chat_sft \
        --model_tag=$MODEL_FOLDER \
        --run=${MODEL_FOLDER}-sft-${NEXT_STEP}
    
    
    echo ""
    echo -e "${GREEN}✓ Completed cycle: ${NEXT_STEP} steps + mid + sft${NC}"
    
    # Move to next chunk
    CURRENT_STEP=$NEXT_STEP
done

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}✓ Pipeline complete!${NC}"
echo -e "${BLUE}Final base training: ${END_STEP} steps${NC}"
echo -e "${BLUE}Model folder: ${MODEL_FOLDER}${NC}"
echo -e "${BLUE}Model path: ~/.cache/nanochat/${MODEL_FOLDER}/${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "To chat with your model:"
echo "  python -m scripts.chat_cli --model_tag=$MODEL_FOLDER"
echo ""
echo "Or serve it on the web:"
echo "  python -m scripts.chat_web --model_tag=$MODEL_FOLDER"
