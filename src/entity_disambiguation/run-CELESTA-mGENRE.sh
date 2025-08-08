#!/bin/bash

# Stop script on error
set -e

# Optional: activate environment
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate celesta

# === Parse input arguments ===
INPUT_FILE=$1

# === Set default values if not provided ===
: "${INPUT_FILE:?test_set.json}"


echo "🔁 Step 1: Converting input JSON to CSV..."
python CELESTA-mGENRE/run_converting_input.py --input_file "$INPUT_FILE" --output_file converted_mentions.csv

echo "🤖 Step 2: Running mGENRE prediction..."
python CELESTA-mGENRE/run_prediction.py --input_csv converted_mentions.csv --output_csv mgenre_predictions.csv

echo "📊 Step 3: Evaluating Entity Disambiguation (ED)..."
python CELESTA-mGENRE/eval_ed.py \
  --pred_file mgenre_predictions.csv \
  --gold_file converted_mentions.csv \
  --output_merged ED_eval_result_merged.csv \
  --output_scores ED_eval_scores.txt

echo "✅ All steps completed successfully."