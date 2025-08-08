import json
import pandas as pd
import argparse

# ==== Argument Parser ====
parser = argparse.ArgumentParser(description="Convert JSON mentions to CSV with marked mentions.")
parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
parser.add_argument("--output_file", type=str, required=True, help="Path to the output CSV file.")
args = parser.parse_args()

# ==== Load JSON ====
with open(args.input_file, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# ==== Convert ====
rows = []
for sent_id, item in enumerate(data, start=1):
    text = item["text"]
    for span in item["spans"]:
        start = span["start"]
        length = span["length"]
        mention = text[start:start+length]

        # Extract Wikidata ID (remove URI part)
        gold_wikidata_id = span["uris"][0].split("/")[-1]

        # Create marked sentence
        marked_sentence = text[:start] + "[START] " + mention + " [END]" + text[start+length:]

        rows.append({
            "sent_id": sent_id,
            "converted_text": marked_sentence,
            "mention": mention,
            "gold_wikidata_id": gold_wikidata_id
        })

# ==== Save to CSV ====
df = pd.DataFrame(rows)
df.to_csv(args.output_file, index=False, encoding="utf-8")

print(f"✅ Conversion completed. Saved to {args.output_file}")
