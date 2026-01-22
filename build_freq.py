import json
import os
import pickle as pkl
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
import config

# Build the token frequency distribution from the corpus
def build_frequency_distribution():
    print(f"Counting word frequency based on corpus...")
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True
    )
        
    vocab_size = tokenizer.vocab_size
    print(f"Model: {config.MODEL_PATH}, Vocab size: {vocab_size}")

    # 2. Initialize count array
    freq_dis = np.zeros(vocab_size, dtype=np.int64)
    
    # 3. Check corpus
    if not os.path.exists(config.FREQ_CORPUS_PATH):
        print("Corpus not found, exiting...")
        exit(0)

    # 4. Scan corpus
    print(f"Scanning: {config.FREQ_CORPUS_PATH}")
    total_files = 0
    with open(config.FREQ_CORPUS_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            try:
                text = line.strip()
                if not text: continue
                
                # Get Token IDs
                input_ids = tokenizer.encode(text, add_special_tokens=False)
                
                # Count
                for idx in input_ids:
                    if idx < vocab_size:
                        freq_dis[idx] += 1
                total_files += 1
            except:
                continue

    print(f"Counting complete, processed {total_files} lines.")

    # 5. Save results
    with open(config.FREQ_FILE, "wb") as f:
        pkl.dump(freq_dis, f)
    
    print(f"Raw counts saved to: {config.FREQ_FILE}")
    '''
    Array index = Token ID
    Array value = Occurrence count of the token in corpus
    '''

if __name__ == "__main__":
    build_frequency_distribution()