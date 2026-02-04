#!/bin/bash

# Hyperparameter search script for BERT4Rec
# Usage: ./run_hyperparam_search.sh [dataset] [n_trials] [num_users]

# Default values
DATASET=${1:-"games"}  # games, baby, or science
N_TRIALS=${2:-25}
NUM_USERS=${3:-2000}

# Map dataset names to file paths
case $DATASET in
    "games")
        DATA_PATH="data/games/amazon_games_user_item_image_text.csv"
        ;;
    "baby")
        DATA_PATH="data/baby/amazon_baby_products_user_item_image_text.csv"
        ;;
    "science")
        DATA_PATH="data/science/amazon_science_user_item_image.csv"
        ;;
    *)
        echo "Unknown dataset: $DATASET"
        echo "Available datasets: games, baby, science"
        exit 1
        ;;
esac

# Check if data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Data file not found at $DATA_PATH"
    exit 1
fi

echo "=========================================="
echo "BERT4Rec Hyperparameter Search"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Data path: $DATA_PATH"
echo "Number of trials: $N_TRIALS"
echo "Number of users: $NUM_USERS"
echo "=========================================="
echo ""

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Change to bert4rec directory and run hyperparameter search
cd bert4rec
python hyperparam.py \
    --data_path "../$DATA_PATH" \
    --n_trials $N_TRIALS \
    --num_users $NUM_USERS \
    --seed 42

echo ""
echo "=========================================="
echo "Hyperparameter search completed!"
echo "Check the hyperparam_search_* directory for results"
echo "=========================================="
