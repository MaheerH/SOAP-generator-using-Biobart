# Medical Chat Summarization using BioBART

## Project Overview

This project focuses on **automatic summarization of medical dialogues** into structured **SOAP notes** (Subjective, Objective, Assessment, Plan).  
The system transforms multi-turn doctor–patient conversations into concise medical summaries suitable for clinical documentation.

The goal was to explore **biomedical domain-specific transformers** and evaluate how fine-tuning impacts their summarization ability.  

The pipeline was developed and tested on the **UIU Medical Dialogue Dataset**, using **BioBART** as the base model.

---

## Thought Process & Approach

### 1. Problem Understanding
Medical dialogues are lengthy, context-rich, and contain both conversational and technical elements.  
The challenge was to extract **clinically relevant information** and express it in a **SOAP-style structured summary**.

### 2. Model Selection
I First performed **token analysis** to estimate average input lengths.  
<p align="center">
  <img src="content/input_lens.jpeg" alt="Input Length Distribution" width="45%"/>
  <img src="content/target_lens.jpeg" alt="Target Length Distribution" width="45%"/>
</p>

This revealed that several samples exceeded 512 tokens, which immediately limited models like **mT5** or **T5-base**.

I selected **BioBART (`GanjinZero/biobart-base`)** because:
- It allows **up to 1,024 input tokens**, providing sufficient headroom.
- It’s trained on **PubMed and clinical corpora**, making it domain-adapted.
- It maintains compatibility with Hugging Face’s `Seq2SeqTrainer`.

### 3. Baseline Evaluation
Before fine-tuning, the base model was tested on the **UIU test set** to establish a performance baseline using ROUGE metrics.

### 4. Fine-Tuning Strategy
Fine-tuning was done using the **Seq2SeqTrainer** from the `transformers` library with:
- Learning rate: `3e-5`
- Batch size: `4`
- Epochs: `3`
- Gradient accumulation: `4`
- Evaluation at every epoch using **ROUGE-1, ROUGE-2, and ROUGE-L**

After this initial run some parameters were updated.
On the second run:
- Learning rate: `4e-5`
- Batch size: `4`
- Epochs: `5`
- Number of beams = `6`

Training was performed on a **Kaggle GPU (P100)**.  
Validation-based checkpoint was implemented to ensure stable convergance.

### 5. Complexity & Key Challenges
- Handling variable-length dialogues required **dynamic tokenization**.
- version mismatch and importing model from huggingface
- Mixed data formats (`.csv` and `.xlsx`) necessitated **flexible loading and validation**.
- GPU memory limits on Kaggle required **gradient accumulation**.

---

## Setup Instructions

### Installation
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

### Run Locally
```bash
python app.py
```


### Requirements
- Python 3.10+
- PyTorch
- Hugging Face Transformers
- Datasets, Evaluate, NLTK, Matplotlib, Seaborn

---

## Model Information

| Component | Details |
|------------|----------|
| Base Model | **GanjinZero/BioBART-Base** |
| Architecture | Seq2Seq Transformer (Encoder–Decoder) |
| Input Max Length | 980 tokens |
| Output Max Length | 650 tokens |
| Domain | Biomedical / Clinical |
| Libraries | `transformers`, `datasets`, `evaluate`, `nltk` |

The model takes raw medical dialogues as input and generates SOAP-formatted summaries.

---

## Fine-Tuning Process

1. **Data Preparation:**
   - Merged `.csv` and `.xlsx` files for train, validation, and test.
   - Ensured presence of `dialogue` and `soap` columns.
   - Removed missing or empty entries.

2. **Tokenization & Collation:**
   - Used `BartTokenizer`.
   - Implemented custom `preprocess_function` to tokenize both dialogue and SOAP text.
   - Used `DataCollatorForSeq2Seq` for dynamic padding.

3. **Training Parameters:**
   - Learning rate: 4e-5  
   - Epochs: 5  
   - Weight decay: 0.01  
   - Gradient accumulation: 4  


4. **Training Curves:**
   4. **Training Curves:**

<p align="center">
  <img src="content/training_loss_curves.png" alt="Training Loss Curve" width="70%">
</p>

   The loss consistently decreased across epochs, indicating stable optimization.

---

## Evaluation Results

### Baseline (BioBART Pre-trained)
| Metric | Score |
|---------|--------|
| ROUGE-1 | 0.49 |
| ROUGE-2 | 0.23 |
| ROUGE-L | 0.30 |

### Fine-Tuned (5 Epochs)
| Metric | Score |
|---------|--------|
| ROUGE-1 | 0.67 |
| ROUGE-2 | 0.40 |
| ROUGE-L | 0.48 |

<p align="center">
  <img src="content/rouge_comparison.png" alt="Training Loss Curve" width="70%">
</p>  

The fine-tuned model achieved:
- **+18.0% improvement in ROUGE-1**
- **+17.1% improvement in ROUGE-2**
- **+18.2% improvement in ROUGE-L**

The generated outputs became more structured and clinically coherent.

<p align="center">
  <img src="content/rouge_progress_curves.png" alt="Rouge progress curve" width="70%">
  <img src="content/metrics_comparison_table.png" alt="Comparing metrics" width="70%">
</p> 


---

## Evaluation Analysis

### ROUGE improvements: 
- ROUGE-2 improved most because fine-tuning enhanced the model’s ability to generate domain-specific, coherent two-word sequences and phrase patterns typical of clinical notes, without necessarily changing word frequency or sentence order drastically.

### Observations:
- The **compression ratio** dropped from 1.28 to 1.01 after fine-tuning → indicating better alignment between generated and reference lengths.
- Fine-tuned outputs captured **SOAP structure** explicitly (e.g., "S:", "O:").
- Reduced hallucinations and improved factual grounding.

### Error Cases:
- Occasional misalignment in “Assessment” section for rare disease mentions.
- Slight redundancy in patient demographic phrases.

<p align="center">
  <img src="content/improvement_comparison.png" alt="Improvement comparison" width="70%">
</p>

---

## 🧰 API Usage Guide

If deployed as an API (for example, via your Hugging Face Space), a typical usage is:

```python
from transformers import BartForConditionalGeneration, BartTokenizer

tokenizer = BartTokenizer.from_pretrained("your-username/biobart-medical-finetuned")
model = BartForConditionalGeneration.from_pretrained("your-username/biobart-medical-finetuned")

dialogue = """Doctor: Hello, how are you feeling today?
Patient: I've been having chest pain for two days."""
inputs = tokenizer(dialogue, return_tensors="pt", truncation=True, max_length=1024)
summary_ids = model.generate(inputs["input_ids"], max_length=650, num_beams=4)
print(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
```

Expected output:
```
S: Patient reports chest pain for two days.
O: No fever or cough.
A: Likely angina.
P: ECG and cardiac enzymes ordered.
```

---

## 🧾 Results Summary

| Stage | ROUGE-1 | ROUGE-2 | ROUGE-L | Improvement |
|--------|----------|----------|----------|--------------|
| Baseline | 0.4943 | 0.2305 | 0.3028 | — |
| Fine-tuned (5 Epochs) | 0.6746 | 0.4015 | 0.4850 | +18–17% |
| Avg Lengths | Ref: 246 words, Gen: 249 words |  |  | Compression ≈ 1.0 |

---

## Contributors
- **Your Name** – Model design, training, and documentation  
- **Contributor 1** – Evaluation and visualization  
- **Contributor 2** – App deployment (Hugging Face Space)

---

## 🔗 Live Demo
➡️ [Visit the Hugging Face Space](https://huggingface.co/spaces/your-username/your-space-name)



