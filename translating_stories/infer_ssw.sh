#!/bin/sh
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1 --ntasks=10 --gres=gpu:l40s:4
#SBATCH --time=48:00:00
#SBATCH --job-name="BatchTranslation"

# Paths and parameters
MODEL="/scratch/nvcela001/submission_contrast/models/trans_deepwide_dp020_african26_7g106_erm3_ft30k"
SRC_LANG="eng"
TGT_LANG="ssw"
BATCH_SIZE=64
INPUT_FILE="/scratch/nvcela001/submission_contrast/data/output_stories/stories_part_3.txt"
OUT_DIR="/scratch/nvcela001/submission_contrast/output/ssw_stories"
TEMP_DIR="/scratch/nvcela001/submission_contrast/temp_split_ssw"

# Create output and temporary directories if they don't exist
mkdir -p "$OUT_DIR"
mkdir -p "$TEMP_DIR"

# Split the input file into stories based on <|endoftext|> delimiter
echo "Splitting input into individual stories..."
csplit -z "$INPUT_FILE" '/<|endoftext|>/' '{*}' --prefix="$TEMP_DIR/story_" --suffix-format="%03d.en"

# Remove <|endoftext|> markers from each story
echo "Cleaning individual stories..."
for STORY_FILE in "$TEMP_DIR"/story_*.en; do
    sed -i '/<|endoftext|>/d' "$STORY_FILE"
done

# Number of stories per batch
STORIES_PER_BATCH=1000  # Adjust as needed

# Create an array of story files
STORY_FILES=("$TEMP_DIR"/story_*.en)

# Total number of stories
TOTAL_STORIES=${#STORY_FILES[@]}

# Calculate the number of batches
NUM_BATCHES=$(( (TOTAL_STORIES + STORIES_PER_BATCH - 1) / STORIES_PER_BATCH ))

# Create batches
echo "Creating batches..."
for (( i=0; i<$NUM_BATCHES; i++ )); do
    START_INDEX=$(( i * STORIES_PER_BATCH ))
    BATCH_FILES=("${STORY_FILES[@]:$START_INDEX:$STORIES_PER_BATCH}")
    BATCH_NAME=$(printf "batch_%03d.en" "$i")
    # Insert empty lines between stories
    for STORY in "${BATCH_FILES[@]}"; do
        cat "$STORY" >> "$TEMP_DIR/$BATCH_NAME"
        echo "" >> "$TEMP_DIR/$BATCH_NAME"
    done
done

# Translate each batch
echo "Translating batches..."
ABS_OUT_DIR=$(realpath "$OUT_DIR")
ABS_MODEL=$(realpath "$MODEL")
for BATCH_FILE in "$TEMP_DIR"/batch_*.en; do
    BASENAME=$(basename "$BATCH_FILE" .en)
    
    echo "Processing batch: $BASENAME"
    
    INPUT_FILE_PATH=$(realpath "$BATCH_FILE")
    OUTPUT_FILE="${BASENAME}.ssw"
    
    # Run the translation script
    sh ./inference_afr.sh "$ABS_MODEL" "$SRC_LANG" "$TGT_LANG" "$BATCH_SIZE" "$INPUT_FILE_PATH" "$ABS_OUT_DIR" "$OUTPUT_FILE"
    
    # Append the batch's translations to the final output file
    echo "Appending translations for batch: $BASENAME"
    cat "$OUT_DIR/${OUTPUT_FILE}" >> "$OUT_DIR/translated_stories.ssw"
done

# Insert <|endoftext|> markers after each story
echo "Inserting <|endoftext|> markers..."
sed -i '/^$/d' "$OUT_DIR/translated_stories.ssw"  # Remove any empty lines
awk 'NR > 1 {print "<|endoftext|>"} {print}' "$OUT_DIR/translated_stories.ssw" > "$OUT_DIR/temp_translated_stories.ssw"
mv "$OUT_DIR/temp_translated_stories.ssw" "$OUT_DIR/translated_stories.ssw"

# Cleanup temporary files
echo "Cleaning up temporary files..."
rm -r "$TEMP_DIR"

echo "Batch translation complete! Output saved to $OUT_DIR/translated_stories.ssw"
