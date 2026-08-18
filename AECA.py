import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import pickle as pkl
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import config

sns.set_theme(style="whitegrid", context="talk", font_scale=1.2)

# Load model, tokenizer, and frequency table resources
def load_resources():
    print(f"Loading model {config.MODEL_PATH} and frequency data {config.FREQ_FILE}...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_threshold=6.0,
        llm_int8_enable_fp32_cpu_offload=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.eval()
    
    with open(config.FREQ_FILE, "rb") as f:
        raw_counts = pkl.load(f)
    
    fre_dis_npy = np.array(raw_counts)
    total = np.sum(fre_dis_npy) + len(fre_dis_npy)
    freq_log_probs = np.log(fre_dis_npy + 1) - np.log(total)
    i_ref_table = - freq_log_probs 
        
    return model, tokenizer, i_ref_table

# Load and parse the labeled dataset from a JSONL file
def load_labeled_dataset(path):
    print(f"Loading labeled dataset: {path}")
    data_items = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    text = item.get('input', '')
                    label = item.get('label')
                    if text and label is not None:
                        data_items.append({'text': text, 'label': label})
                except: continue
    except:
        return []
    
    print(f"Loaded {len(data_items)} samples")
    return data_items

# Apply dynamic entropic convolution using a difference operator on the potential stream
def apply_dynamic_entropic_convolution(potential_stream):
    shock_spectrum = np.empty_like(potential_stream)
    shock_spectrum[:-1] = potential_stream[:-1] - potential_stream[1:]
    shock_spectrum[-1] = potential_stream[-1]
    return shock_spectrum

# Generate Static Loss Stream (NLL) and Information Potential Stream (p * I) for a given text
def calculate_streams(text, model, tokenizer, i_ref_table, temperature=1.0):
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=1024,
        padding=False  
    ).to(config.DEVICE)
    input_ids = inputs['input_ids'][0]
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, :-1, :]
        labels = input_ids[1:]
        
        if temperature != 1.0:
            logits = logits / temperature
        
        log_probs = torch.log_softmax(logits, dim=-1)
        target_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        nll_seq = (-target_log_probs).cpu().numpy() 
        
        target_probs = torch.gather(torch.softmax(logits, dim=-1), dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        p_theta = target_probs.cpu().numpy()  
        
        labels_np = labels.cpu().numpy()
        max_vocab = len(i_ref_table)  
        
        out_of_bounds_mask = labels_np >= max_vocab
        clipped_labels = np.clip(labels_np, 0, max_vocab - 1)
        i_ref_seq = i_ref_table[clipped_labels]
        
        default_i_ref = 0.5
        i_ref_seq[out_of_bounds_mask] = default_i_ref
        
        potential_stream = p_theta * i_ref_seq

    return nll_seq, potential_stream, p_theta

# Calculate metrics from scores and labels
def get_metrics(scores, labels):
    scores = np.array(scores)
    labels = np.array(labels)
    
    valid_mask = np.isfinite(scores) & np.isfinite(labels)
    clean_scores = scores[valid_mask]
    clean_labels = labels[valid_mask]
    fpr_list, tpr_list, thresholds = roc_curve(clean_labels, clean_scores)
    auroc = auc(fpr_list, tpr_list)

    return auroc

# Evaluate
def evaluate_aeca(data_items, model, tokenizer, i_ref_table, output_dir):
    print("Starting AECA (Adaptive Entropic Convolutional Analysis)...")

    sample_core_data = []
    valid_labels = []
    
    for item in tqdm(data_items, desc="Preprocessing samples"):
        text = item['input']
        label = item['label']
        if len(text) < 10:
            continue
        
        nll_seq, potential_stream, _ = calculate_streams(text, model, tokenizer, i_ref_table)
        reconstructed_turbulence = apply_dynamic_entropic_convolution(potential_stream)
        
        std_turbulence = np.std(reconstructed_turbulence)
        std_nll = np.std(nll_seq)
        
        sample_core_data.append({
            'std_turbulence': std_turbulence,
            'std_nll': std_nll
        })
        valid_labels.append(label)
    
    all_results = []
    best_auroc = 0.0
    result = None
    
    for llambda in range(1, 11):
        y_scores = []
        for core_data in sample_core_data:
            score = core_data['std_turbulence'] - llambda * core_data['std_nll']
            y_scores.append(score)
        
        auroc = get_metrics(y_scores, valid_labels)
        
        current_result = {
            'llambda': llambda,
            'auroc': f"{auroc:.2%}",
        }
        all_results.append(current_result)
        
        current_auroc = float(current_result['auroc'].strip('%')) / 100
        if current_auroc > best_auroc:
            best_auroc = current_auroc
            result = current_result
    
    print(f"AUROC: {result['auroc']}")

# Main
if __name__ == "__main__":
    model, tokenizer, i_ref_table = load_resources()
    DATASET_PATH = config.DATASET_PATH 

    data = []
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"Loaded {len(data)} samples from {DATASET_PATH}")
    
    evaluate_aeca(data, model, tokenizer, i_ref_table, config.OUTPUT_DIR)