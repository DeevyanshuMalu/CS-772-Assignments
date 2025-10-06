import torch
from transformers import pipeline
import json
import os
import argparse
from tqdm import tqdm
from prompts import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--temp", type=float, default=1.0)
parser.add_argument("--top_p", type=float, default=1.0)

args = parser.parse_args()
temp = args.temp
top_p = args.top_p

with open("../hin/hin_test.jsonl", "r") as f:
    data = [json.loads(line) for line in f.readlines()]

model_id = "meta-llama/Llama-3.2-1B-Instruct"
model_id = "meta-llama/Llama-3.1-8B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

actuals = []
preds = []

for item in tqdm(data, desc="Testing", total=len(data)):
    src = item["english word"]
    act = item["native word"]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.format(input=src)},
    ]

    outputs = pipe(
        messages,
        max_new_tokens=256,
        temperature=temp,
        top_p=top_p,
        return_full_text=False,
    )
    pred = outputs[0]["generated_text"].strip()

    actuals.append(act)
    preds.append(pred)

# print("Actual:", actuals)
# print("Predicted:", preds)

accuracy = compute_accuracy(actuals, preds)
f1 = get_f1_score(actuals, preds)

os.makedirs("results", exist_ok=True)
with open(
    os.path.join("results", f"test_results_{temp}temp_{top_p}top_p.txt"), "w"
) as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
