DATASET_ROOT=$1

# Define scenes
SCENES=("bicycle" "bonsai" "counter" "garden" "kitchen" "room" "stump")

echo "=========================================="
echo "Starting Process"
echo "Dataset Root   : $DATASET_ROOT"
echo "=========================================="

# 4. Loop through scenes
for scene in "${SCENES[@]}"; do
    echo "Processing scene: $scene..."

    CURRENT_DATA_PATH="$DATASET_ROOT/$scene"
    OUTPUT_PATH="outputs/varying/$scene"

    python run_iir_train.py \
        -s "$CURRENT_DATA_PATH" \
        --eval \
        -m "$OUTPUT_PATH" \
        --target_exp 0.3

    echo "Finished: $scene"
    echo "------------------------------------------"
done

echo "All tasks completed successfully."