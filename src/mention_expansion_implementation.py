# Apply mention expansion for github repo of celesta

import argparse
import os
import pandas as pd
import json

def parse_arguments():
    parser = argparse.ArgumentParser(description="Apply mention expansions to the sentences")
    parser.add_argument('--prompt_type', type=str, default='few-shot',
                        help='Prompt type (few-shot | zero-shot)')
    parser.add_argument('--dataset', type=str, default='IndGEL',
                        help='Dataset name (e.g., IndGEL, IndQEL, IndEL_Wikidata)')
    parser.add_argument('--llm1', type=str, default='Llama-3',
                        help='First LLM name (e.g., Llama-3)')
    parser.add_argument('--llm2', type=str, default='Komodo',
                        help='Second LLM name (e.g., Komodo)')

    parser.add_argument('--expansion_base', type=str, default='../mention_expansion_results',
                        help='Base dir where the TSV expansion file lives')
    parser.add_argument('--original_json_base', type=str, default='../ReFinED_format',
                        help='Base dir to the original ReFinED-format JSON files')
    parser.add_argument('--output_base', type=str, default='../with_mention_expansion',
                        help='Base dir for writing the updated JSON with mention expansion')

    return parser.parse_args()


def main():
    args = parse_arguments()

    # --- Paths ---
    expansion_path = (
        f"{args.expansion_base}/{args.dataset}/{args.prompt_type}/"
        f"selected_expansion_{args.llm1}_{args.llm2}_{args.prompt_type}_{args.dataset}.tsv"
    )
    output_json_path = (
        f"{args.output_base}/{args.dataset}/{args.prompt_type}/"
        f"test_set_with_{args.llm1}_{args.llm2}_{args.dataset}_{args.prompt_type}.json"
    )
    original_json_path = f"{args.original_json_base}/test_set_{args.dataset}.json"

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    # --- Load expansions ---
    expanded_mentions = {}
    with open(expansion_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for num, line in enumerate(lines):
            if num == 0:
                continue
            items = line.strip().split("\t")
            if len(items) >= 4:
                mention = items[3].strip()
                if mention.lower() == "none" or mention == "":
                    expanded_mentions[num] = None
                else:
                    expanded_mentions[num] = mention
            else:
                expanded_mentions[num] = None

    # --- Load original JSONL ---
    with open(original_json_path, "r", encoding="utf-8") as f:
        original_docs = [json.loads(line) for line in f]

    # --- Apply expansions ---
    final_docs = []
    mention_index = 1

    for num, doc in enumerate(original_docs):
        text = doc["text"]
        spans = doc.get("spans", [])
        new_spans = []
        offset = 0

        for span in spans:
            original_start = span["start"]
            original_length = span["length"]
            updated_start = original_start + offset

            original_mention = text[updated_start:updated_start + original_length]
            expanded_mention = expanded_mentions[mention_index]

            mention_to_use = (
                expanded_mention.strip()
                if isinstance(expanded_mention, str) and expanded_mention.strip()
                else original_mention
            )
            new_length = len(mention_to_use)

            # Replace mention
            text = text[:updated_start] + mention_to_use + text[updated_start + original_length:]

            # Add updated span
            new_spans.append({
                "start": updated_start,
                "length": new_length,
                "uris": span["uris"]
            })

            offset += new_length - original_length
            mention_index += 1

        doc["text"] = text
        doc["spans"] = new_spans
        final_docs.append(doc)

    # --- Save final output ---
    with open(output_json_path, "w", encoding="utf-8") as f:
        for doc in final_docs:
            json.dump(doc, f, ensure_ascii=False)
            f.write("\n")

    print(f"Mention expansion implementation is saved to: {output_json_path}")


if __name__ == "__main__":
    main()
