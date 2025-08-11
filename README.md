# CELESTA

![GitHub license](https://img.shields.io/github/license/dice-group/CELESTA)
![GitHub stars](https://img.shields.io/github/stars/dice-group/CELESTA?style=social)

CELESTA is a hybrid Entity Disambiguation (ED) framework designed for low-resource languages. In a case study on Indonesian, CELESTA performs parallel mention expansion using both multilingual and monolingual Large Language Models (LLMs). It then applies a similarity-based selection mechanism to choose the expansion that is most semantically aligned with the original context. Finally, the selected expansion is linked to a knowledge base entity using an off-the-shelf ED model—without requiring any fine-tuning. The following is the architecture of CELESTA:

<p align="center">
<img src="images/celesta_architecture.png" width="95%">
</p>


## 📂 Repository Structure
```
│
├── datasets/                     	# Input datasets (IndGEL, IndQEL, IndEL-WIKI)
├── images/                       	# Architecture visualizations
│   └── celesta_architecture.jpg
├── src/                          	# Source code for CELESTA modules
│   └── mention_expansion/        	# Mention expansion scripts
│   └── mention_expansion_selection/    # Mention expansion selection scripts
├── requirements.txt              	# Python dependencies
├── README.md                     	# Project overview
└── LICENSE                       	# License file
```

## ⚙️ Installation

1. **Clone the repository**
```bash
   
   git clone https://github.com/dice-group/CELESTA.git
   cd CELESTA 

```

2. **Create the environment**
```

conda create -n celesta python=3.10
conda activate celesta
pip install -r requirements.txt

```
3. **Install CELESTA-mGENRE**
```

# change folder to entity_disambiguation directory
cd entity_disambiguation

# run script to install CELESTA-mGENRE
bash INSTALL-CELESTA-mGENRE.sh

```

## Evaluation

### 📊 Datasets

CELESTA is evaluated on three Indonesian Entity Disambiguation (ED) datasets: **IndGEL**, **IndQEL**, and **IndEL-WIKI**.  
- **IndGEL** (general domain) and **IndQEL** (specific domain) are from the [IndEL dataset](https://github.com/dice-group/IndEL).  
- **IndEL-WIKI** is a new dataset we created to provide additional evaluation data for CELESTA.

| Dataset Property             | IndGEL | IndQEL | IndEL-WIKI |
|------------------------------|-------:|-------:|-----------:|
| **Sentences**                | 2,114  | 2,621  | 24,678     |
| **Total entities**           | 4,765  | 2,453  | 24,678     |
| **Unique entities**          | 55     | 16     | 24,678     |
| **Entities / sentence**      | 2.4    | 1.6    | 1.0        |
| **Train set sentences**      | 1,674  | 2,076  | 17,172     |
| **Validation set sentences** | 230    | 284    | 4,958      |
| **Test set sentences**       | 230    | 284    | 4,958      |



### 🤖 Large Language Models (LLMs)

CELESTA uses **two hybrid LLMs**:

- **Multilingual LLMs**
  - [LLaMA-3](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)
  - [Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)

- **Indonesian Monolingual LLMs**
  - [Komodo](https://huggingface.co/suayptalha/Komodo-7B-Instruct)
  - [Merak](https://huggingface.co/Ichsan2895/Merak-7B-v4-GGUF)

## 🚀 Usage
### Mention Expansion
1. Run Mention Expansion
```
# Change directory to the src folder
cd src

# To run the mention expansion script
# usage: mention_expansion.py [-h] [--model_name MODEL_NAME] [--prompt_type PROMPT_TYPE] [--dataset DATASET] [--split SPLIT] [--llm_name LLM_NAME] [--input_dir INPUT_DIR]
#                            [--output_dir OUTPUT_DIR] [--batch_size BATCH_SIZE] [--save_every SAVE_EVERY] [--save_interval SAVE_INTERVAL]

python mention_expansion.py --model_name meta-llama/Meta-Llama-3-70B-Instruct --prompt_type few-shot --dataset IndGEL --llm_name llama-3

```

2. Entity Disambiguation
### Entity Disambiguation with mGENRE
```
# Run script to CELESTA-mGENRE
bash run-CELESTA-mGENRE.sh
```

### 📈 Results
1. General Performance

The table below compares CELESTA with two baseline ED models ([ReFinED](https://github.com/amazon-science/ReFinED) and [mGENRE](https://github.com/facebookresearch/GENRE)) across the three evaluation datasets. **Bold** values indicate the highest score for each metric within a dataset.


| Dataset     | Model            | Precision | Recall  | F1      |
|-------------|------------------|-----------|---------|---------|
| **IndGEL**  | ReFinED          | **0.749**     | 0.547   | 0.633   |
|             | mGENRE           | 0.742     | 0.718   | 0.730   |
|             | **CELESTA (ours)** | 0.748 | **0.722** | **0.735** |
| **IndQEL**  | ReFinED          | 0.208     | 0.160   | 0.181   |
|             | mGENRE           | 0.298 | 0.298 | 0.298 |
|             | **CELESTA (ours)** | **0.298** | **0.298** | **0.298** |
| **IndEL-WIKI** | ReFinED       | **0.627** | 0.327   | 0.430   |
|             | mGENRE           | 0.601     | 0.489 | 0.539 |
|             | **CELESTA (ours)** | 0.595     | **0.495**   | **0.540**   |


The table below reports Precision (P), Recall (R), and F1 for CELESTA and individual LLM configurations across the three datasets, under both **zero-shot** and **few-shot** prompting. **Bold** values mark the highest F1 score within each dataset–prompting combination. Results are shown for CELESTA using **ReFinED** to generate candidate entities and retrieve the corresponding Wikidata URIs.

<table>
<thead>
<tr>
<th rowspan="2">Dataset</th>
<th rowspan="2">Model</th>
<th colspan="3">Zero-shot</th>
<th colspan="3">Few-shot</th>
</tr>
<tr>
<th>P</th><th>R</th><th>F1</th>
<th>P</th><th>R</th><th>F1</th>
</tr>
</thead>
<tbody>

<!-- IndGEL -->
<tr>
<td rowspan="9"><b>IndGEL</b></td>
<td>LLaMA-3</td><td>0.727</td><td><b>0.499</b></td><td><b>0.592</b></td><td>0.777</td><td>0.531</td><td>0.631</td>
</tr>
<tr><td>Mistral</td><td>0.699</td><td>0.411</td><td>0.517</td><td><b>0.806</b></td><td>0.310</td><td>0.448</td></tr>
<tr><td>Komodo</td><td>0.709</td><td>0.447</td><td>0.548</td><td>0.704</td><td>0.527</td><td>0.603</td></tr>
<tr><td>Merak</td><td>0.654</td><td>0.441</td><td>0.526</td><td>0.749</td><td>0.547</td><td>0.633</td></tr>

<tr style="background-color:#f0f0f0">
<td colspan="7"><b>CELESTA with ReFinED</b></td>
</tr>

<tr><td>LLaMA-3 & Komodo</td><td><b>0.731</b></td><td>0.437</td><td>0.547</td><td>0.757</td><td>0.513</td><td>0.612</td></tr>
<tr><td>LLaMA-3 & Merak</td><td>0.688</td><td>0.431</td><td>0.530</td><td>0.802</td><td><b>0.586</b></td><td><b>0.677</b></td></tr>
<tr><td>Mistral & Komodo</td><td>0.719</td><td>0.390</td><td>0.506</td><td>0.781</td><td>0.344</td><td>0.478</td></tr>
<tr><td>Mistral & Merak</td><td>0.678</td><td>0.402</td><td>0.505</td><td>0.779</td><td>0.503</td><td>0.611</td></tr>

<!-- IndQEL -->
<tr>
<td rowspan="9"><b>IndQEL</b></td>
<td>LLaMA-3</td><td>0.154</td><td>0.051</td><td>0.077</td><td><b>0.327</b></td><td>0.058</td><td>0.099</td>
</tr>
<tr><td>Mistral</td><td>0.179</td><td>0.131</td><td>0.151</td><td>0.072</td><td>0.029</td><td>0.042</td></tr>
<tr><td>Komodo</td><td>0.158</td><td>0.116</td><td>0.134</td><td>0.208</td><td><b>0.160</b></td><td><b>0.181</b></td></tr>
<tr><td>Merak</td><td><b>0.203</b></td><td><b>0.149</b></td><td><b>0.172</b></td><td>0.142</td><td>0.106</td><td>0.121</td></tr>

<tr style="background-color:#f0f0f0">
<td colspan="7"><b>CELESTA with ReFinED</b></td>
</tr>

<tr><td>LLaMA-3 & Komodo</td><td>0.138</td><td>0.047</td><td>0.071</td><td>0.282</td><td>0.073</td><td>0.116</td></tr>
<tr><td>LLaMA-3 & Merak</td><td>0.160</td><td>0.113</td><td>0.132</td><td>0.130</td><td>0.098</td><td>0.112</td></tr>
<tr><td>Mistral & Komodo</td><td>0.138</td><td>0.095</td><td>0.112</td><td>0.107</td><td>0.047</td><td>0.066</td></tr>
<tr><td>Mistral & Merak</td><td>0.196</td><td>0.146</td><td>0.167</td><td>0.128</td><td>0.095</td><td>0.109</td></tr>

<!-- IndEL-WIKI -->
<tr>
<td rowspan="9"><b>IndEL-WIKI</b></td>
<td>LLaMA-3</td><td>0.581</td><td>0.234</td><td>0.332</td><td>0.639</td><td>0.322</td><td>0.428</td>
</tr>
<tr><td>Mistral</td><td>0.565</td><td>0.232</td><td>0.329</td><td>0.552</td><td>0.201</td><td>0.294</td></tr>
<tr><td>Komodo</td><td>0.592</td><td>0.256</td><td>0.357</td><td>0.591</td><td>0.270</td><td>0.370</td></tr>
<tr><td>Merak</td><td>0.591</td><td><b>0.285</b></td><td><b>0.385</b></td><td>0.548</td><td>0.293</td><td>0.382</td></tr>

<tr style="background-color:#f0f0f0">
<td colspan="7"><b>CELESTA with ReFinED</b></td>
</tr>

<tr><td>LLaMA-3 & Komodo</td><td>0.577</td><td>0.234</td><td>0.332</td><td>0.639</td><td>0.322</td><td>0.428</td></tr>
<tr><td>LLaMA-3 & Merak</td><td><b>0.596</b></td><td>0.273</td><td>0.374</td><td><b>0.641</b></td><td><b>0.355</b></td><td><b>0.457</b></td></tr>
<tr><td>Mistral & Komodo</td><td>0.576</td><td>0.231</td><td>0.330</td><td>0.575</td><td>0.219</td><td>0.317</td></tr>
<tr><td>Mistral & Merak</td><td>0.564</td><td>0.248</td><td>0.345</td><td>0.581</td><td>0.270</td><td>0.369</td></tr>

</tbody>
</table>

These results show CELESTA’s performance when using mGENRE for candidate generation and Wikidata URI retrieval.

<table>
<thead>
<tr>
<th rowspan="2">Dataset</th>
<th rowspan="2">Model</th>
<th colspan="3">Zero-shot</th>
<th colspan="3">Few-shot</th>
</tr>
<tr>
<th>P</th><th>R</th><th>F1</th>
<th>P</th><th>R</th><th>F1</th>
</tr>
</thead>
<tbody>

<!-- IndGEL -->
<tr>
<td rowspan="9"><b>IndGEL</b></td>
<td>LLaMA-3</td><td><b>0.720</b></td><td><b>0.694</b></td><td><b>0.707</b></td><td>0.742</td><td>0.718</td><td>0.730</td>
</tr>
<tr><td>Mistral</td><td>0.667</td><td>0.640</td><td>0.653</td><td>0.607</td><td>0.584</td><td>0.595</td></tr>
<tr><td>Komodo</td><td>0.702</td><td>0.668</td><td>0.685</td><td>0.740</td><td>0.698</td><td>0.718</td></tr>
<tr><td>Merak</td><td>0.611</td><td>0.576</td><td>0.594</td><td>0.696</td><td>0.672</td><td>0.684</td></tr>

<tr style="background-color:#f0f0f0">
<td colspan="7"><b>CELESTA with mGENRE</b></td>
</tr>

<tr><td>LLaMA-3 & Komodo</td><td>0.695</td><td>0.660</td><td>0.677</td><td>0.741</td><td>0.708</td><td>0.724</td></tr>
<tr><td>LLaMA-3 & Merak</td><td>0.631</td><td>0.596</td><td>0.613</td><td><b>0.748</b></td><td><b>0.722</b></td><td><b>0.735</b></td></tr>
<tr><td>Mistral & Komodo</td><td>0.657</td><td>0.632</td><td>0.644</td><td>0.623</td><td>0.602</td><td>0.612</td></tr>
<tr><td>Mistral & Merak</td><td>0.620</td><td>0.588</td><td>0.603</td><td>0.702</td><td>0.676</td><td>0.686</td></tr>

<!-- IndQEL -->
<tr>
<td rowspan="9"><b>IndQEL</b></td>
<td>LLaMA-3</td><td>0.298</td><td>0.298</td><td>0.298</td><td><b>0.274</b></td><td><b>0.273</b></td><td><b>0.273</b></td>
</tr>
<tr><td>Mistral</td><td>0.258</td><td>0.258</td><td>0.258</td><td>0.185</td><td>0.182</td><td>0.183</td></tr>
<tr><td>Komodo</td><td>0.252</td><td>0.251</td><td>0.251</td><td>0.269</td><td>0.269</td><td>0.269</td></tr>
<tr><td>Merak</td><td>0.233</td><td>0.233</td><td>0.233</td><td>0.255</td><td>0.255</td><td>0.255</td></tr>

<tr style="background-color:#f0f0f0">
<td colspan="7"><b>CELESTA with mGENRE</b></td>
</tr>

<tr><td>LLaMA-3 & Komodo</td><td><b>0.298</b></td><td><b>0.298</b></td><td><b>0.298</b></td><td>0.266</td><td>0.266</td><td>0.266</td></tr>
<tr><td>LLaMA-3 & Merak</td><td>0.276</td><td>0.276</td><td>0.276</td><td>0.0.256</td><td>0.255</td><td>0.255</td></tr>
<tr><td>Mistral & Komodo</td><td>0.262</td><td>0.262</td><td>0.262</td><td>0.185</td><td>0.182</td><td>0.183</td></tr>
<tr><td>Mistral & Merak</td><td>0.236</td><td>0.236</td><td>0.236</td><td>0.202</td><td>0.200</td><td>0.201</td></tr>

<!-- IndEL-WIKI -->
<tr>
<td rowspan="9"><b>IndEL-WIKI</b></td>
<td>LLaMA-3</td><td>0.516</td><td><b>0.415</b></td><td>0.460</td><td>0.601</td><td>0.489</td><td>0.539</td>
</tr>
<tr><td>Mistral</td><td>0.457</td><td>0.360</td><td>0.403</td><td>0.447</td><td>0.363</td><td>0.401</td></tr>
<tr><td>Komodo</td><td>0.542</td><td>0.401</td><td>0.461</td><td>0.547</td><td>0.422</td><td>0.476</td></tr>
<tr><td>Merak</td><td>0.474</td><td>0.371</td><td>0.417</td><td>0.428<td>0.353</td><td>0.387</td></tr>

<tr style="background-color:#f0f0f0">
<td colspan="7"><b>CELESTA with mGENRE</b></td>
</tr>

<tr><td>LLaMA-3 & Komodo</td><td><b>0.548</b></td><td>0.411</td><td><b>0.470</b></td><td><b>0.618</b></td><td>0.481</td><td>0.537</td></tr>
<tr><td>LLaMA-3 & Merak</td><td>0.521</td><td>0.412</td><td>0.460</td><td>0.595</td><td><b>0.495</b></td><td><b>0.540</b></td></tr>
<tr><td>Mistral & Komodo</td><td>0.500</td><td>0.368</td><td>0.424</td><td>0.484</td><td>0.382</td><td>0.427</td></tr>
<tr><td>Mistral & Merak</td><td>0.447</td><td>0.349</td><td>0.392</td><td>0.507</td><td>0.413</td><td>0.455</td></tr>

</tbody>
</table>

2. Contribution of LLMs to CELESTA’s Correct Predictions

In addition to overall performance, we measure the contribution of each multilingual and monolingual LLM, as well as the original mention, to CELESTA’s correct predictions in the dual multilingual–monolingual mention expansion setup, using IndGEL with few-shot prompting. A contribution is counted when a mention expansion (or the original mention) is selected by CELESTA through its similarity-based selection mechanism and leads to a correct entity prediction. The table below reports contributions when CELESTA uses either ReFinED or mGENRE for candidate generation and Wikidata URI retrieval. Values indicate the percentage of correct predictions attributed to LLM1 (multilingual), LLM2 (monolingual), or the original mention for each LLM pair. These results highlight the complementary strengths of multilingual and monolingual LLMs and the benefit of pairing them with high-recall ED backends.

| LLM Pair          | LLM1 (%)  | LLM2 (%) | Original (%) |
|-------------------|-----------|----------|--------------|
| **CELESTA with ReFinED**                                |
| LLaMA-3 & Komodo  | **64.49** | 12.32    | 23.19        |
| LLaMA-3 & Merak   | 41.46     | **56.71**| 1.83         |
| Mistral & Komodo  | **79.28** | 16.22    | 4.5          |
| Mistral & Merak   | 43.24     | **56.76**| 0.0          |
| **CELESTA with mGENRE**                                 |
| LLaMA-3 & Komodo  | **59.83** | 14.53    | 25.64        |
| LLaMA-3 & Merak   | 41.62     | **57.54**| 1.12	  |
| Mistral & Komodo  | **78.86** | 14.43    | 6.38         |
| Mistral & Merak   | 46.87     | **53.43**| 0.0          |


## 📫 Contact

If you have any questions or feedbacks, feel free to contact us at ria.hari.gusmita@uni-paderborn.de or ria.gusmita@uinjkt.ac.id

