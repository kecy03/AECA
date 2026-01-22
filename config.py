import torch
import os

# Model path
MODEL_PATH = "pythia-1.4b" 

# Corpus path for token frequency statistics
FREQ_CORPUS_PATH = "c4.json" 

# File to save token frequency counts
model_id = os.path.basename(MODEL_PATH)  
OUTPUT_DIR = f"{model_id}_token_frequency"
FREQ_FILE = os.path.join(OUTPUT_DIR, "c4.pkl") 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Path to the dataset for evaluation
DATASET_PATH = "pubmed_central.json"