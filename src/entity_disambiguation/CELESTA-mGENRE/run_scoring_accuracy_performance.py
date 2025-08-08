import pandas as pd

# === Step 1: Load prediction and gold files ===
df_pred = pd.read_csv("mgenre_predictions.csv", dtype=str).fillna("")
df_gold = pd.read_csv("converted_mentions.csv", dtype=str).fillna("")

# === Step 2: Rename columns for consistency ===
df_pred.rename(columns={"wikidata_id": "wikidata_id"}, inplace=True)
df_gold.rename(columns={"gold_wikidata_id": "wikidata_id"}, inplace=True)

# === Step 3: Normalize Wikidata IDs (strip prefixes if needed) ===
# def normalize_qid(qid):
#     qid = qid.strip()
#     if qid.startswith("http"):
#         return qid.split("/")[-1]
#     return qid

df_pred["wikidata_id"] = df_pred["wikidata_id"]#.apply(normalize_qid)
df_gold["wikidata_id"] = df_gold["wikidata_id"]#.apply(normalize_qid)

# === Step 4: Merge prediction and gold by sent_id and mention ===
# Remove duplicates in predictions based on sent_id + mention
df_pred_unique = df_pred.drop_duplicates(subset=["sent_id", "mention"], keep="first")

# Merge without causing row multiplication
merged = pd.merge(
    df_gold,
    df_pred_unique[["sent_id", "mention", "wikidata_id"]],
    on=["sent_id", "mention"],
    how="left",
    suffixes=("_gold", "_pred")
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
merged.to_csv("ED_eval_result_merged.csv", index=False)

with open("ED_eval_scores.txt", "w") as f:
    f.write(f"Correct predictions: {correct_preds}\n")
    f.write(f"Total predictions:   {total_predicted}\n")
    f.write(f"Total gold entries:  {total_gold}\n")
    f.write(f"Precision:           {precision:.4f}\n")
    f.write(f"Recall:              {recall:.4f}\n")
    f.write(f"F1 Score:            {f1:.4f}\n")
