import os

import datasets
# todo: rename to datasets.py?
import torch
from datasets import load_dataset, load_from_disk
from transformers import LlamaTokenizerFast

HF_DATASET = "monology/pile-uncopyrighted"

TOKENIZER_PATH = "./"
DATASET_PATH = "./data/pile_uncopyrighted.hf"
TOKENIZED_DATASET_PATH = "./data/pile_small_tokenized_2b_reshaped.pt"

# from neel
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

tokenizer = LlamaTokenizerFast.from_pretrained(TOKENIZER_PATH)


def tokenize_and_chunk(text):
    return tokenizer(
        text,
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )


# we get a dict with a text field, containing a list of strings
# for each string, we should tokenize it, then chunk it into 128 token chunks, then return a dict with a tokens field,
# containing a flattened list of tokenized chunks
def batched_tokenize_and_chunk(rows):
    return {
        "tokens": [
            tokens
            for text in rows["text"]
            for tokens in tokenize_and_chunk(text)["input_ids"]
        ]
    }


def fetch_dataset():
    if os.path.exists(TOKENIZED_DATASET_PATH):
        print("Found local tokenized dataset, loading...")
        tokenized_dataset = torch.load(TOKENIZED_DATASET_PATH)
    else:
        if os.path.exists(DATASET_PATH):
            print("Found local dataset, loading...")
            raw_dataset = load_from_disk(DATASET_PATH)
        else:
            print("No local dataset found, fetching...")
            raw_dataset = load_dataset(HF_DATASET, split="train[:1000000]", cache_dir="./cache")
            print("Saving dataset...")
            raw_dataset.save_to_disk(DATASET_PATH)

        print("Setting dataset format...")  # from neel's code, sets the dataset format to torch, not sure if needed
        raw_dataset.set_format(type="torch", columns=["text"])
        print("Tokenizing dataset...")
        tokenized_dataset = raw_dataset.map(
            batched_tokenize_and_chunk,
            batched=True,
            num_proc=None,
            remove_columns=raw_dataset.column_names,
        )

        print("Saving tokenized dataset...")
        torch.save(tokenized_dataset, TOKENIZED_DATASET_PATH)

    return tokenized_dataset.shuffle()


data = fetch_dataset()
