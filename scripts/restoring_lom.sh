IMAGE_TYPE=$1
DATASET_ROOT=$2

# Define scenes
SCENES=("bike" "buu" "chair" "shrub" "sofa")

echo "=========================================="
echo "Starting Process"
echo "Image Type  : $IMAGE_TYPE"
echo "Dataset Root: $DATASET_ROOT"
echo "=========================================="

# Loop through scenes
for scene in "${SCENES[@]}"; do
    echo "Processing scene: $scene..."

    CURRENT_DATA_PATH="$DATASET_ROOT/$scene"
    OUTPUT_PATH="outputs/$IMAGE_TYPE/$scene"

    EXTRA_ARGS=""
    if [ "$scene" == "bike" ]; then
        EXTRA_ARGS="--stage 5"
    fi

    python run_iir_train.py \
        -s "$CURRENT_DATA_PATH" \
        --eval \
        -m "$OUTPUT_PATH" \
        --images "$IMAGE_TYPE" \
        $EXTRA_ARGS

    echo "Finished: $scene"
    echo "------------------------------------------"
done

echo "All tasks completed successfully."