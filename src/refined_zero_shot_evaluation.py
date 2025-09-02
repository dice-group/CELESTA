import argparse
import json

from refined.inference.processor import Refined
from refined.evaluation.evaluation import evaluate_on_docs
from refined.data_types.doc_types import Doc, Span
from refined.data_types.base_types import Entity


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate CELESTA with ReFinED in the zero-shot setting.")
    parser.add_argument('--input_dir', type=str, default="../CELESTA/with_mention_expansion",
                        help="Base directory containing the JSON inputs")
    parser.add_argument('--dataset', type=str, default="IndGEL",
                        help="Dataset name (e.g., IndGEL, IndQEL, IndEL-Wiki)")
    parser.add_argument('--prompt_type', type=str, default="few-shot",
                        help="Prompt type used to produce the expansions (few-shot | zero-shot)")
    parser.add_argument('--llm1', type=str, required=True,
                        help="First LLM name used in the pair (e.g., Llama-3)")
    parser.add_argument('--llm2', type=str, required=True,
                        help="Second LLM name used in the pair (e.g., Komodo)")

    # ReFinED-related
    parser.add_argument('--model_name', type=str, default='wikipedia_model_with_numbers',
                        help="ReFinED model name")
    parser.add_argument('--entity_set', type=str, default='wikidata',
                        help="Entity set for ReFinED (wikidata or wikipedia)")
    parser.add_argument('--use_precomputed_descriptions', action='store_true',
                        help="Use precomputed descriptions in ReFinED")
    parser.add_argument('--ed_threshold', type=float, default=0.15,
                        help="ED threshold passed to evaluate_on_docs")

    return parser.parse_args()


def load_indel_docs(json_path):
    docs = []
    with open(json_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            example = json.loads(line)
            spans = []
            for s in example.get("spans", []):
                start = s["start"]
                length = s["length"]
                mention_text = example["text"][start:start + length]
                qid = s["uris"][0].split("/")[-1] if s.get("uris") else None

                # Create the span
                span = Span(
                    text=mention_text,
                    start=start,
                    ln=length,
                    gold_entity=Entity(wikidata_entity_id=qid)
                )

                # Initialize empty candidate list
                span.candidate_entities = []
                spans.append(span)

            docs.append(Doc(
                doc_id=f"indel-{i}",
                text=example["text"],
                tokens=example["text"].split(),
                spans=spans,
                md_spans=[]
            ))
    return docs


def main():
    args = parse_arguments()

    # Build input JSON path (split fixed as 'test_set')
    json_path = (
        f"{args.input_dir}/"
        f"{args.dataset}/"
        f"{args.prompt_type}/"
        f"test_set_with_mention_expansion_{args.llm1}_{args.llm2}.json"
    )

    # Load ReFinED model
    refined = Refined.from_pretrained(
        model_name=args.model_name,
        entity_set=args.entity_set,
        use_precomputed_descriptions=args.use_precomputed_descriptions
    )

    # Load docs
    indel_docs = load_indel_docs(json_path)

    print(
        f"Evaluating CELESTA on {args.dataset}, with selected expansion from "
        f"{args.llm1}-{args.llm2} using {args.prompt_type} prompting"
    )

    # ED only — el=False
    metrics = evaluate_on_docs(
        refined=refined,
        docs=indel_docs,
        dataset_name=args.dataset,
        ed_threshold=args.ed_threshold,
        el=False
    )

    # Print F1 summary
    print(metrics.get_summary())


if __name__ == "__main__":
    main()
