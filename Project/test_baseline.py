import os
from datasets import load_from_disk
from transformers import AutoTokenizer, BartForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig
from torch.utils.data import DataLoader
import torch
import argparse
import pickle
from tqdm import tqdm
import numpy as np
from prompt import PROMPT
import json


data_path = "paradetox_dataset"

ds = load_from_disk(data_path)

base_model_name = "facebook/bart-base"
model_name = "s-nlp/bart-base-detox"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)


with open("results/baseline.jsonl", "w") as f:
    for row in tqdm(ds["test"], total=len(ds["test"])):
        input_ids = tokenizer.encode(row["en_toxic_comment"], return_tensors="pt")
        output_ids = model.generate(input_ids, max_length=50, num_return_sequences=1)
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        f.write(
            json.dumps(
                {
                    "hate": row["en_toxic_comment"],
                    "pred": output_text,
                    "actual": row["en_neutral_comment"],
                }
            )
            + "\n"
        )
