# Expand the original mention of sentences using multilingual and monolingual LLMs

import argparse
import torch
import pandas as pd
import time
import os
import ast
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoConfig,
    BitsAndBytesConfig,
)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Mention Expansion with Komodo")
    parser.add_argument('--model_name', type=str, default="suayptalha/Komodo-7B-Instruct")
    parser.add_argument('--prompt_type', type=str, default="zero-shot")
    parser.add_argument('--dataset', type=str, default="IndGEL")
    parser.add_argument('--split', type=str, default="test")
    parser.add_argument('--llm_name', type=str, default="Komodo")
    parser.add_argument('--input_dir', type=str, default="../datasets")
    parser.add_argument('--output_dir', type=str, default="../mention_expansion_results")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--save_every', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=300)
    return parser.parse_args()

class MentionDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return row["sent_id"], row["sentence"], row["mention"]

def format_prompt(mention, sentence, prompt_type="zero-shot"):
    if prompt_type == "zero-shot":
        return f"""You are given an entity mention and a sentence containing it. Your task is to expand the mention based on contextual clues in the sentence.

        Mention: \"{mention}\"  
        Sentence: \"{sentence}\"
        
        Instructions:
        - Expand the mention in **Indonesian**, using clues from the sentence.
        - If it is an **abbreviation or acronym**, return its full form.
        - If it is an **incomplete name** (e.g., only a first name or nickname), return the full name.
        - If it is already complete, return: Response= Complete
        - If you cannot find a suitable expansion, return: Response= None
        - Only return the expansion in this format:
        
        Response="""
    elif prompt_type == "few-shot":
        few_shot_examples = """**Example Outputs:**
        - Input: mention = "PBB", sentence = "Indonesia adalah anggota aktif PBB sejak awal."  
          Response= Perserikatan Bangsa-Bangsa
        - Input: mention = "UI", sentence = "Dia lulus dari UI dengan predikat cumlaude."  
          Response= Universitas Indonesia
        - Input: mention = "Joko", sentence = "Presiden Joko menghadiri KTT G20 di Bali."  
          Response= Joko Widodo
        - Input: mention = "Bung Karno", sentence = "Bung Karno dikenal sebagai proklamator kemerdekaan Indonesia."  
          Response= Soekarno
        - Input: mention = "Presiden Prabowo Subianto", sentence = "Presiden Prabowo Subianto akan mengunjungi Malaysia bulan ini."  
          Response= Complete"""

        
        task_instruction = f"""
        Prompt:
        Expand the following entity mention \"{mention}\" based on the context: \"{sentence}\".
        
        **Instructions:**
        - Provide a meaningful expansion of the given mention in **Indonesian**, based on the provided context.
        - If the mention is an **abbreviation or acronym**, return its full form.
        - If the mention is an **incomplete name** (e.g., only a first name or partial name), return the full name.
        - If the mention is already complete (a full name or a full term), return: Response= Complete
        - If no suitable expansion is found, return: Response= None.
        - Only return the response in the format:
        
        Response="""
        return few_shot_examples + "\n\n" + task_instruction
    else:    
        return f"""Provide the full form of the following entity mention based on the sentence context. If no expansion can be found, return: Response= None

        Entity: "{mention}"
        Sentence: "{sentence}"
        
        Response="""

def extract_response(text):    
    if "Response=" in text:
        response = text.split("Response=")[-1].strip()
        response = response.split("\n")[0].strip() 
        response = response.split("You are given")[0].strip()
        response = response.split(". ")[0].strip()
        response = response.replace("[", "").replace("]", "").replace("\"", "").strip()
        if response.lower() == "none" or response == "":
            return "None"
        return response
    return "None"


def load_existing_results(path):
    if os.path.exists(path):
        df = pd.read_csv(path, sep="\t")        
        return df.to_dict(orient="records"), set(df["sent_id"])
    return [], set()

def save_results(results, path):
    pd.DataFrame(results).to_csv(path, sep="\t", index=False)    

def expand_mentions(model, tokenizer, device, dataloader, prompt_type, output_path, already_done, save_every, save_interval):
    results, _ = load_existing_results(output_path)
    with torch.inference_mode():
        count = len(results)
        last_save_time = time.time()
        progress_bar = tqdm(total=len(dataloader.dataset), desc="Expanding Mentions")

        for batch in dataloader:
            sent_ids, sentences, mentions = batch
            for sid, sent, mention in zip(sent_ids, sentences, mentions):
                progress_bar.update(1)

                if sid in already_done:
                    continue

                prompt = format_prompt(mention, sent, prompt_type)

                if prompt is None or not isinstance(prompt, str) or prompt.strip() == "":
                    raise ValueError(f"Generated prompt is invalid or empty for mention '{mention}' (sid: {sid}).")             

                inputs = tokenizer(text=prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=False,
                    temperature=0.1,
                    eos_token_id=tokenizer.eos_token_id
                )
                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                decoded = decoded.replace(prompt, "").strip()
                response = extract_response(decoded)

                results.append({
                    "sent_id": sid,
                    "sentence": sent,
                    "mention": mention,
                    "full_response": decoded,
                    "mention expansion": response
                })

                count += 1
                if count % save_every == 0 or (time.time() - last_save_time > save_interval):
                    save_results(results, output_path)
                    last_save_time = time.time()

        progress_bar.close()
        save_results(results, output_path)
    return results

def main():
    args = parse_arguments()
    start_time = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    config = AutoConfig.from_pretrained(args.model_name)
    if config.model_type in ["mt5", "t5", "marian", "mbart"]:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

    df_path = f'{args.input_dir}/{args.dataset}/test_set.tsv'
    df = pd.read_csv(df_path, sep="\t")

    save_dir = f'{args.output_dir}/{args.dataset}/{args.prompt_type}/resultsUsing{args.llm_name}/'
    os.makedirs(save_dir, exist_ok=True)
    output_path = f'{save_dir}/entity_expansion_{args.dataset}_{args.llm_name}.tsv'

    already_done_df, already_done_ids = load_existing_results(output_path)

    dataset = MentionDataset(df)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    expand_mentions(
        model, tokenizer, device, dataloader, args.prompt_type,
        output_path, already_done_ids, args.save_every, args.save_interval
    )

    elapsed_time = time.time() - start_time
    print(f"Finished in {elapsed_time // 60:.0f} minutes {elapsed_time % 60:.2f} seconds")

if __name__ == "__main__":
    main()
