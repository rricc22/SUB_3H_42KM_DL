#!/usr/bin/env bash
# Complete training pipeline for heart rate prediction models.
#
# This script runs data preprocessing and model training with full control over:
# - Number of samples to process
# - Which model(s) to train
# - All hyperparameters
#
# Usage:
#     bash run_pipeline.sh [OPTIONS]
#
# Examples:
#     bash run_pipeline.sh                          # Process ALL data, train both models
#     bash run_pipeline.sh -n 10000 -m lstm         # 10K samples, train basic LSTM only
#     bash run_pipeline.sh -s -m both -e 50         # Skip preprocessing, train both with 50 epochs

set -e  # Exit on error

# ═══════════════════════════════════════════════════════════════════════════
# DEFAULT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Preprocessing
MAX_SAMPLES="None"              # None = ALL data (~253K workouts)
SKIP_PREPROCESSING=false

# Model selection
MODEL="both"                    # Options: lstm, lstm_embeddings, both

# Training hyperparameters
EPOCHS=100
BATCH_SIZE=32
LR=0.001
HIDDEN_SIZE=64
NUM_LAYERS=2
DROPOUT=0.2
PATIENCE=10
DEVICE="auto"

# Paths
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${PROJECT_DIR}/DATA/processed"
PREPROCESSING_SCRIPT="${PROJECT_DIR}/Preprocessing/prepare_sequences_v2.py"
TRAIN_SCRIPT="${PROJECT_DIR}/Model/train.py"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints"

# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

show_help() {
    cat << EOF
Usage: run_pipeline.sh [OPTIONS]

Complete pipeline for heart rate prediction: preprocessing + training

PREPROCESSING OPTIONS:
  -n, --max-samples N       Number of samples to preprocess (default: None = ALL ~253K)
  -s, --skip-preprocessing  Skip data preprocessing step

MODEL SELECTION:
  -m, --model MODEL         Model to train: lstm, lstm_embeddings, or both (default: both)

TRAINING HYPERPARAMETERS:
  -e, --epochs N            Number of training epochs (default: 100)
  -b, --batch-size N        Batch size (default: 32)
  --lr RATE                 Learning rate (default: 0.001)
  --hidden-size N           LSTM hidden dimension (default: 64)
  --num-layers N            Number of LSTM layers (default: 2)
  --dropout RATE            Dropout probability (default: 0.2)
  --patience N              Early stopping patience (default: 10)
  --device DEVICE           Device: cuda, cpu, or auto (default: auto)

OTHER:
  -h, --help                Show this help message

EXAMPLES:
  # Process ALL data and train both models (full pipeline)
  bash run_pipeline.sh

  # Quick test with 1,000 samples, train only basic LSTM
  bash run_pipeline.sh -n 1000 -m lstm -e 20

  # Use 50,000 samples, train both models
  bash run_pipeline.sh -n 50000 -e 100

  # Skip preprocessing (data already exists), train LSTM with embeddings
  bash run_pipeline.sh -s -m lstm_embeddings

  # Custom hyperparameters
  bash run_pipeline.sh -n 10000 -e 50 -b 16 --lr 0.0005 --hidden-size 128

EOF
}

print_header() {
    local title="$1"
    echo ""
    echo "================================================================================"
    echo "$title"
    echo "================================================================================"
    echo ""
}

print_step() {
    local step="$1"
    local total="$2"
    local title="$3"
    echo ""
    echo "================================================================================"
    echo "[$step/$total] $title"
    echo "================================================================================"
}

print_success() {
    echo "✓ $1"
}

print_error() {
    echo "✗ ERROR: $1" >&2
}

# ═══════════════════════════════════════════════════════════════════════════
# PARSE COMMAND-LINE ARGUMENTS
# ═══════════════════════════════════════════════════════════════════════════

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -n|--max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        -s|--skip-preprocessing)
            SKIP_PREPROCESSING=true
            shift
            ;;
        -m|--model)
            MODEL="$2"
            if [[ ! "$MODEL" =~ ^(lstm|lstm_embeddings|both)$ ]]; then
                print_error "Invalid model '$MODEL'. Choose: lstm, lstm_embeddings, or both"
                exit 1
            fi
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --hidden-size)
            HIDDEN_SIZE="$2"
            shift 2
            ;;
        --num-layers)
            NUM_LAYERS="$2"
            shift 2
            ;;
        --dropout)
            DROPOUT="$2"
            shift 2
            ;;
        --patience)
            PATIENCE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ═══════════════════════════════════════════════════════════════════════════
# DISPLAY CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

print_header "HEART RATE PREDICTION - TRAINING PIPELINE"

echo "CONFIGURATION:"
echo "  Preprocessing:"
if [ "$MAX_SAMPLES" = "None" ]; then
    echo "    Max samples:        ALL (~253K workouts)"
else
    echo "    Max samples:        $MAX_SAMPLES"
fi
echo "    Skip preprocessing: $SKIP_PREPROCESSING"
echo ""
echo "  Model Selection:"
case "$MODEL" in
    lstm)
        echo "    Training:           Basic LSTM only"
        ;;
    lstm_embeddings)
        echo "    Training:           LSTM with embeddings only"
        ;;
    both)
        echo "    Training:           Both models (Basic LSTM + LSTM with embeddings)"
        ;;
esac
echo ""
echo "  Training Hyperparameters:"
echo "    Epochs:             $EPOCHS"
echo "    Batch size:         $BATCH_SIZE"
echo "    Learning rate:      $LR"
echo "    Hidden size:        $HIDDEN_SIZE"
echo "    Num layers:         $NUM_LAYERS"
echo "    Dropout:            $DROPOUT"
echo "    Patience:           $PATIENCE"
echo "    Device:             $DEVICE"
echo ""

# Determine total steps
if [ "$SKIP_PREPROCESSING" = true ]; then
    if [ "$MODEL" = "both" ]; then
        TOTAL_STEPS=3  # Train LSTM + Train Embeddings + Summary
    else
        TOTAL_STEPS=2  # Train Model + Summary
    fi
else
    if [ "$MODEL" = "both" ]; then
        TOTAL_STEPS=4  # Preprocess + Train LSTM + Train Embeddings + Summary
    else
        TOTAL_STEPS=3  # Preprocess + Train Model + Summary
    fi
fi

CURRENT_STEP=0

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: PREPROCESSING (optional)
# ═══════════════════════════════════════════════════════════════════════════

if [ "$SKIP_PREPROCESSING" = false ]; then
    CURRENT_STEP=$((CURRENT_STEP + 1))
    print_step "$CURRENT_STEP" "$TOTAL_STEPS" "PREPROCESSING DATA"
    
    # Check if preprocessing script exists
    if [ ! -f "$PREPROCESSING_SCRIPT" ]; then
        print_error "Preprocessing script not found: $PREPROCESSING_SCRIPT"
        exit 1
    fi
    
    # Backup and modify MAX_SAMPLES in preprocessing script
    BACKUP_FILE="${PREPROCESSING_SCRIPT}.bak"
    cp "$PREPROCESSING_SCRIPT" "$BACKUP_FILE"
    
    if [ "$MAX_SAMPLES" = "None" ]; then
        print_success "Processing ALL workouts from endomondoHR.json"
        sed -i "s/MAX_SAMPLES = [0-9]\+/MAX_SAMPLES = None/" "$PREPROCESSING_SCRIPT"
    else
        print_success "Processing first $MAX_SAMPLES workouts from endomondoHR.json"
        sed -i "s/MAX_SAMPLES = [0-9]\+/MAX_SAMPLES = $MAX_SAMPLES/" "$PREPROCESSING_SCRIPT"
        sed -i "s/MAX_SAMPLES = None/MAX_SAMPLES = $MAX_SAMPLES/" "$PREPROCESSING_SCRIPT"
    fi
    
    # Run preprocessing
    echo ""
    python3 "$PREPROCESSING_SCRIPT"
    PREPROCESS_EXIT_CODE=$?
    
    # Restore original file
    mv "$BACKUP_FILE" "$PREPROCESSING_SCRIPT"
    
    if [ $PREPROCESS_EXIT_CODE -ne 0 ]; then
        print_error "Preprocessing failed with exit code $PREPROCESS_EXIT_CODE"
        exit 1
    fi
    
    # Verify data was created
    if [ ! -f "${DATA_DIR}/train.pt" ] || [ ! -f "${DATA_DIR}/val.pt" ] || [ ! -f "${DATA_DIR}/test.pt" ]; then
        print_error "Preprocessing completed but data files not found in $DATA_DIR"
        exit 1
    fi
    
    print_success "Preprocessing completed successfully"
else
    echo ""
    print_success "Skipping preprocessing (using existing data)"
    
    # Verify data exists
    if [ ! -f "${DATA_DIR}/train.pt" ] || [ ! -f "${DATA_DIR}/val.pt" ] || [ ! -f "${DATA_DIR}/test.pt" ]; then
        print_error "Data files not found in $DATA_DIR. Run without -s to preprocess data first."
        exit 1
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

train_model() {
    local model_name="$1"
    local model_display_name="$2"
    
    CURRENT_STEP=$((CURRENT_STEP + 1))
    print_step "$CURRENT_STEP" "$TOTAL_STEPS" "TRAINING $model_display_name"
    
    # Check if training script exists
    if [ ! -f "$TRAIN_SCRIPT" ]; then
        print_error "Training script not found: $TRAIN_SCRIPT"
        exit 1
    fi
    
    print_success "Starting training..."
    echo ""
    
    # Run training
    python3 "$TRAIN_SCRIPT" \
        --model "$model_name" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --hidden_size "$HIDDEN_SIZE" \
        --num_layers "$NUM_LAYERS" \
        --dropout "$DROPOUT" \
        --patience "$PATIENCE" \
        --device "$DEVICE" \
        --data_dir "$DATA_DIR" \
        --checkpoint_dir "$CHECKPOINT_DIR"
    
    TRAIN_EXIT_CODE=$?
    
    if [ $TRAIN_EXIT_CODE -ne 0 ]; then
        print_error "Training $model_display_name failed with exit code $TRAIN_EXIT_CODE"
        exit 1
    fi
    
    echo ""
    print_success "$model_display_name training completed"
    print_success "Checkpoint saved to: ${CHECKPOINT_DIR}/${model_name}_best.pt"
    print_success "Training curves saved to: ${CHECKPOINT_DIR}/${model_name}_training_curves.png"
}

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2/3: TRAIN SELECTED MODEL(S)
# ═══════════════════════════════════════════════════════════════════════════

case "$MODEL" in
    lstm)
        train_model "lstm" "BASIC LSTM"
        ;;
    lstm_embeddings)
        train_model "lstm_embeddings" "LSTM WITH EMBEDDINGS"
        ;;
    both)
        train_model "lstm" "BASIC LSTM"
        train_model "lstm_embeddings" "LSTM WITH EMBEDDINGS"
        ;;
esac

# ═══════════════════════════════════════════════════════════════════════════
# FINAL STEP: SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

CURRENT_STEP=$((CURRENT_STEP + 1))
print_step "$CURRENT_STEP" "$TOTAL_STEPS" "RESULTS SUMMARY"

echo "OUTPUT FILES:"
case "$MODEL" in
    lstm)
        if [ -f "${CHECKPOINT_DIR}/lstm_best.pt" ]; then
            print_success "checkpoints/lstm_best.pt"
            print_success "checkpoints/lstm_training_curves.png"
        fi
        ;;
    lstm_embeddings)
        if [ -f "${CHECKPOINT_DIR}/lstm_embeddings_best.pt" ]; then
            print_success "checkpoints/lstm_embeddings_best.pt"
            print_success "checkpoints/lstm_embeddings_training_curves.png"
        fi
        ;;
    both)
        if [ -f "${CHECKPOINT_DIR}/lstm_best.pt" ]; then
            print_success "checkpoints/lstm_best.pt"
            print_success "checkpoints/lstm_training_curves.png"
        fi
        if [ -f "${CHECKPOINT_DIR}/lstm_embeddings_best.pt" ]; then
            print_success "checkpoints/lstm_embeddings_best.pt"
            print_success "checkpoints/lstm_embeddings_training_curves.png"
        fi
        ;;
esac

echo ""
echo "Next steps:"
echo "  1. Review training curves in checkpoints/*.png"
echo "  2. Check model performance (MAE < 10 BPM is acceptable, < 5 BPM is excellent)"
echo "  3. If needed, retrain with different hyperparameters"

print_header "✅ PIPELINE COMPLETE!"

exit 0
