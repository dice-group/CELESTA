import pandas as pd
import argparse

# === Argument Parser ===
parser = argparse.ArgumentParser(description="Evaluate Entity Disambiguation (ED) performance.")
parser.add_argument("--pred_file", type=str, required=True, help="Path to the predicted CSV file (e.g., mgenre_predictions.csv)")
parser.add_argument("--gold_file", type=str, required=True, help="Path to the gold standard CSV file (e.g., converted_mentions.csv)")
parser.add_argument("--output_merged", type=str, default="ED_eval_result_merged.csv", help="Path to save the merged output CSV")
parser.add_argument("--output_scores", type=str, default="ED_eval_scores.txt", help="Path to save the evaluation scores")
args = parser.parse_args()

# === Step 1: Load prediction and gold files ===
df_pred = pd.read_csv(args.pred_file, dtype=str).fillna("")
df_gold = pd.read_csv(args.gold_file, dtype=str).fillna("")

# === Step 2: Rename columns for consistency ===
df_gold.rename(columns={"gold_wikidata_id": "wikidata_id_gold"}, inplace=True)
df_pred.rename(columns={"wikidata_id": "wikidata_id_pred"}, inplace=True)

# === Step 3: Normalize Wikidata IDs if needed ===
# df_pred["wikidata_id_pred"] = df_pred["wikidata_id_pred"].apply(normalize_qid)
# df_gold["wikidata_id_gold"] = df_gold["wikidata_id_gold"].apply(normalize_qid)

# === Step 4: Merge prediction and gold by sent_id and mention ===
df_pred_unique = df_pred.drop_duplicates(subset=["sent_id", "mention"], keep="first")

merged = pd.merge(
    df_gold,
    df_pred_unique[["sent_id", "mention", "wikidata_id_pred"]],
    on=["sent_id", "mention"],
    how="left"
)

# === Step 5: Compute exact match column ===
merged["correct"] = (
    merged["wikidata_id_gold"].str.strip().str.lower() ==
    merged["wikidata_id_pred"].str.strip().str.lower()
).astype(int)

# === Step 6: Count for evaluation ===
correct_preds = merged["correct"].sum()
total_predicted = (merged["wikidata_id_pred"] != "").sum()
total_gold = (merged["wikidata_id_gold"] != "").sum()

# === Step 7: Compute metrics ===
precision = correct_preds / total_predicted if total_predicted > 0 else 0
recall = correct_preds / total_gold if total_gold > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

# === Step 8: Print metrics ===
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# === Step 9: Save results ===
merged.to_csv(args.output_merged, index=False)

with open(args.output_scores, "w") as f:
    f.write(f"Correct predictions: {correct_preds}\n")
    f.write(f"Total predictions:   {total_predicted}\n")
    f.write(f"Total gold entries:  {total_gold}\n")
    f.write(f"Precision:           {precision:.4f}\n")
    f.write(f"Recall:              {recall:.4f}\n")
    f.write(f"F1 Score:            {f1:.4f}\n")
