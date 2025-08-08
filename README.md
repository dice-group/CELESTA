# CELESTA

![GitHub license](https://img.shields.io/github/license/dice-group/CELESTA)
![GitHub stars](https://img.shields.io/github/stars/dice-group/CELESTA?style=social)

CELESTA is a hybrid Entity Disambiguation (ED) framework designed for low-resource languages. In a case study on Indonesian, CELESTA performs parallel mention expansion using both multilingual and monolingual Large Language Models (LLMs). It then applies a similarity-based selection mechanism to choose the expansion that is most semantically aligned with the original context. Finally, the selected expansion is linked to a knowledge base entity using an off-the-shelf ED model—without requiring any fine-tuning. The following is the architecture of CELESTA:

<p align="center">
<img src="images/celesta_architecture.jpg" width="75%">
</p>


## 📂 Repository Structure
```
│
├── datasets/                     # Input datasets (IndGEL, IndQEL, IndEL-WIKI)
├── images/                       # Architecture visualizations
│   └── celesta_architecture.jpg
├── src/                          # Source code for CELESTA modules
│   └── mention_expansion/        # Mention expansion scripts
├── requirements.txt              # Python dependencies
├── README.md                     # Project overview
└── LICENSE                       # License file
```

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/dice-group/CELESTA.git
   cd CELESTA ```

2. **Create the environment**
```
conda create -n celesta python=3.10
conda activate celesta
pip install -r requirements.txt
```

## Evaluation Dataset
CELESTA is evaluated on three Indonesian ED datasets, i.e. IndGEL, IndQEL, and IndEL-WIKI. The first two datasets come from [IndEL dataset](https://github.com/dice-group/IndEL) in which IndGEL is the general domain and IndQEL is the specific domain. We created the third dataset, IndEL-WIKI, to provide more datasets to evaluate CELESTA. The followings are the detail of each datasets:

|Dataset's Property           |IndGEL|IndQEL|IndEL-WIKI|
|-----------------------------|------|------|----------|
|Sentences  	              |2114  |2621  |24678     |
|Total entities               |4765  |2453  |24678     |
|Unique entites               |55    |16    |24678     |
|Entities (avg) in a sentence |2.4   |1.6   |1.0       | 
|Sentences in train set       |1674  |2076  |17172     |
|Sentences in val set         |230   |284   |4958      |
|Sentences in test set        |230   |284   |4958      |


## Large Language Models (LLMs)

CELESTA uses **two parallel LLMs**:

- **Multilingual LLMs**
  - [LLaMA-3 (70B)](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)
  - [Mistral (7B)](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)

- **Indonesian Monolingual LLMs**
  - [Komodo (7B)](https://huggingface.co/suayptalha/Komodo-7B-Instruct)
  - [Merak (7B)](https://huggingface.co/Ichsan2895/Merak-7B-v4-GGUF)


## 🚀 Usage

1. Run Mention Expansion
```
# Change directory to the src folder
cd src

# To run the mention expansion script
# usage: mention_expansion.py [-h] [--model_name MODEL_NAME] [--prompt_type PROMPT_TYPE] [--dataset DATASET] [--split SPLIT] [--llm_name LLM_NAME] [--input_dir INPUT_DIR]
#                            [--output_dir OUTPUT_DIR] [--batch_size BATCH_SIZE] [--save_every SAVE_EVERY] [--save_interval SAVE_INTERVAL]

python mention_expansion.py --model_name meta-llama/Meta-Llama-3-70B-Instruct --prompt_type few-shot --dataset IndGEL --llm_name llama-3 
```

