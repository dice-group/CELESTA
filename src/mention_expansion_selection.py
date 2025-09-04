# mention expansion selection from two LLMs results using similarity measurement

import argparse
import torch
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import csv
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="Mention Expansion Similarity-Based Selection")
    parser.add_argument('--input_dir', type=str, default="../mention_expansion_results",
                        help="Base directory containing dataset folder")
    parser.add_argument('--output_dir', type=str, default="../similarity_based_expansion_selection",
                        help="Directory to save the selected results")
    parser.add_argument('--dataset', type=str, default="IndGEL",
                        help="Dataset name, e.g., IndGEL, IndQEL")
    parser.add_argument('--prompt_type', type=str, default="few-shot",
                        help="Prompt type: few-shot or zero-shot")
    parser.add_argument('--threshold', type=float, default=0.80,
                        help="Similarity threshold")
    return parser.parse_args()

def main():
    args = parse_arguments()

    dataset = args.dataset
    prompt = args.prompt_type
    THRESHOLD = args.threshold

    # ==== Paths ====
    input_path = f"{args.input_dir}/{dataset}/{prompt}/mention_expansion_allLLMs_{dataset}.tsv"
    output_dir = f"{args.output_dir}/{dataset}/{prompt}/"
    os.makedirs(output_dir, exist_ok=True)

    # ==== All LLM pairs to test ====
    llm_pairs = [
        ("Llama-3", "Komodo"),
        ("Llama-3", "Merak"),
        ("Mistral", "Komodo"),
        ("Mistral", "Merak")
    ]

    # ==== Load dataset ====
    df = pd.read_csv(input_path, sep="\t", quoting=csv.QUOTE_NONE, engine="python", on_bad_lines="skip")

    # If sent_id not in input, generate one
    if "sent_id" not in df.columns:
        df.insert(0, "sent_id", range(1, len(df) + 1))

    # ==== Load Sentence-BERT model ====
    model = SentenceTransformer("sentence-transformers/LaBSE")
    model.eval()

    # ==== Cache embeddings ====
    embeddings_cache = {}
    expansion_cache = {}

    for idx, row in df.iterrows():
        sentence = row.get("sentence")
        mention = row.get("mention")

        if isinstance(sentence, str) and isinstance(mention, str):
            sentence_with_mention = sentence.replace(mention, f"[{mention}]")
            embeddings_cache[(idx, "original")] = model.encode(sentence_with_mention, convert_to_tensor=True)

            for col in df.columns:
                if col in ["sent_id", "sentence", "mention"]:
                    continue
                if isinstance(row[col], str) and row[col].strip().lower() != "none":
                    expansion_sentence = sentence.replace(mention, row[col].strip())
                    expansion_cache[(idx, col)] = model.encode(expansion_sentence, convert_to_tensor=True)
                else:
                    expansion_cache[(idx, col)] = None

    def process_llm_pair(llm1, llm2):
        results = []
        for idx, row in df.iterrows():
            sent_id = row.get("sent_id")
            sentence = row.get("sentence")
            mention = row.get("mention")

            if not isinstance(sentence, str) or not isinstance(mention, str):
                continue

            emb_mention = embeddings_cache[(idx, "original")]
            routed_llms = [llm1, llm2]
            candidates, similarity_scores, expansions_dict = [], {}, {}

            for llm in routed_llms:
                emb_exp = expansion_cache.get((idx, llm))
                if emb_exp is not None:
                    expansion = row[llm].strip()
                    score = util.pytorch_cos_sim(emb_mention, emb_exp).item()
                    candidates.append((llm, expansion, score))
                    similarity_scores[llm] = round(score, 4)
                    expansions_dict[llm] = expansion
                else:
                    similarity_scores[llm] = None
                    expansions_dict[llm] = None

            rejected_due_to_threshold = False

            if len(candidates) == 1:
                best_llm, best_expansion, best_score = candidates[0]
                if best_score < THRESHOLD:
                    rejected_due_to_threshold = True
                    best_llm, best_expansion, best_score = None, mention, None
            elif len(candidates) > 1:
                best_llm, best_expansion, best_score = max(candidates, key=lambda x: x[2])
                if best_score < THRESHOLD:
                    rejected_due_to_threshold = True
                    best_llm, best_expansion, best_score = None, mention, None
            else:
                best_llm, best_expansion, best_score = None, mention, None
                similarity_scores = {llm: None for llm in routed_llms}

            results.append({
                "sent_id": sent_id,
                "mention": mention,
                "sentence": sentence,
                "routed_llms": ",".join(routed_llms),
                "best_llm": best_llm,
                "best_expansion": best_expansion,
                "best_score": round(best_score, 4) if best_score is not None else None,
                "llm1_name": routed_llms[0],
                "llm1_expansion": expansions_dict.get(routed_llms[0]),
                "llm1_score": similarity_scores.get(routed_llms[0]),
                "llm2_name": routed_llms[1],
                "llm2_expansion": expansions_dict.get(routed_llms[1]),
                "llm2_score": similarity_scores.get(routed_llms[1]),
                "rejected_due_to_threshold": rejected_due_to_threshold
            })

        output_path = os.path.join(output_dir, f"selected_expansion_with_scores_{llm1}_{llm2}_{prompt}_{dataset}.tsv")
        pd.DataFrame(results).to_csv(output_path, sep="\t", index=False)
        print(f"✅ Saved results for {llm1} & {llm2} → {output_path}")

    for llm1, llm2 in llm_pairs:
        process_llm_pair(llm1, llm2)

if __name__ == "__main__":
    main()
