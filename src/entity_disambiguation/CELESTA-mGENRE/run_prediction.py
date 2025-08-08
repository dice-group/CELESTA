import pandas as pd
from collections import defaultdict, Counter
from tqdm import tqdm
import sys
import pickle
import torch
import json
import argparse

from genre.trie import Trie, MarisaTrie
from torch.serialization import add_safe_globals
from omegaconf import DictConfig
from omegaconf.base import ContainerMetadata
from genre.fairseq_model import mGENRE

# ===== Argument Parser =====
parser = argparse.ArgumentParser(description="Run mGENRE predictions on mention-annotated sentences.")
parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file (converted_mentions.csv)")
parser.add_argument("--output_csv", type=str, default="mgenre_predictions.csv", help="Path to the output CSV file")
args = parser.parse_args()

# ===== Setup GPU =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

add_safe_globals([DictConfig, ContainerMetadata])

# ===== Load mappings and trie =====
with open("../data/lang_title2wikidataID-normalized_with_redirect.pkl", "rb") as f:
    lang_title2wikidataID = pickle.load(f)

with open("../data/titles_lang_all105_marisa_trie_with_redirect.pkl", "rb") as f:
    trie = pickle.load(f)

# ===== Patch torch.load =====
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

# ===== Load model and move to GPU =====
model = mGENRE.from_pretrained("../models/fairseq_multilingual_entity_disambiguation").eval().to(device)

# ===== Top prediction filter =====
def filter_top_predictions(predictions, mentions, lang="id"):
    top_by_mention = {}
    flat_preds = predictions[0]

    if len(mentions) == 1:
        lang_filtered = [p for p in flat_preds if p["text"].strip().endswith(f">> {lang}")]
        if lang_filtered:
            best_pred = max(lang_filtered, key=lambda p: p["score"].item() if hasattr(p["score"], "item") else p["score"])
            top_by_mention[mentions[0]] = {
                "mention": mentions[0],
                "text": best_pred["text"],
                "id": best_pred["id"],
                "score": best_pred["score"].item() if hasattr(best_pred["score"], "item") else best_pred["score"]
            }
        else:
            top_by_mention[mentions[0]] = {"mention": mentions[0], "text": "NOT_FOUND", "id": None, "score": -1.0}
        return top_by_mention

    mention_counts = Counter(mentions)
    mention_indices = defaultdict(int)

    for mention in mentions:
        mention_indices[mention] += 1
        mention_key = f"{mention}@{mention_indices[mention]}" if mention_counts[mention] > 1 else mention

        mention_lc = mention.lower()
        best_match, best_score = None, -100.0

        for pred in flat_preds:
            if not pred["text"].strip().endswith(f">> {lang}"):
                continue
            if mention_lc in pred["text"].lower():
                score = pred["score"].item() if hasattr(pred["score"], "item") else pred["score"]
                if score > best_score:
                    best_score = score
                    best_match = {"mention": mention, "text": pred["text"], "id": pred["id"], "score": score}

        if best_match:
            top_by_mention[mention_key] = best_match
        else:
            top_by_mention[mention_key] = {"mention": mention, "text": "NOT_FOUND", "id": None, "score": -1.0}

    return top_by_mention

# ===== Load CSV =====
df = pd.read_csv(args.input_csv)

# ===== Prediction Function =====
def predict_entity(row):
    sent_id = row["sent_id"]
    sentence = row["converted_text"]
    mentions = [row["mention"]]

    try:
        predictions = model.sample(
            sentences=[sentence],
            prefix_allowed_tokens_fn=lambda batch_id, sent: [
                e for e in trie.get(sent.tolist()) if e < len(model.task.target_dictionary)
            ],
            text_to_id=lambda x: max(
                lang_title2wikidataID[tuple(reversed(x.split(" >> ")))],
                key=lambda y: int(y[1:])
            ),
            marginalize=False
        )

        # Serialize all predictions
        all_preds_serialized = json.dumps([
            {
                "text": p["text"],
                "id": p["id"],
                "score": p["score"].item() if hasattr(p["score"], "item") else p["score"]
            }
            for p in predictions[0]
        ], ensure_ascii=False)

        # Get top predictions
        top_results = filter_top_predictions(predictions, mentions, lang="id")
        return [
            {
                "sent_id": sent_id,
                "converted_text": sentence,
                "mention": v["mention"],
                "predicted_entity_text": v["text"],
                "wikidata_id": v["id"],
                "score": v["score"],
                "all_predictions": all_preds_serialized
            }
            for v in top_results.values()
        ]
    except Exception as e:
        return [{
            "sent_id": sent_id,
            "converted_text": sentence,
            "mention": mentions,
            "predicted_entity_text": f"ERROR: {e}",
            "wikidata_id": None,
            "score": None,
            "all_predictions": "[]"
        }]

# ===== Apply Prediction with Progress Bar =====
expanded_rows = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting entities (GPU)"):
    expanded_rows.extend(predict_entity(row))

# ===== Save Result =====
expanded_df = pd.DataFrame(expanded_rows)
expanded_df.to_csv(args.output_csv, index=False)
print(f"✅ Saved to {args.output_csv}")
