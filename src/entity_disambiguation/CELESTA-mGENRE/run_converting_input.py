import json
import pandas as pd

# ==== Config ====
input_file = "test_set.json"  # Path to your file
output_file = "converted_mentions.csv"  # Output CSV file

# ==== Load JSON ====
with open(input_file, "r", encoding="utf-8") as f:
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
df.to_csv(output_file, index=False, encoding="utf-8")

print(f"✅ Conversion completed. Saved to {output_file}")
