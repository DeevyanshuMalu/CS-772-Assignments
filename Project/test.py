import os
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    AutoModel,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig
from torch.utils.data import DataLoader
import torch
import argparse
import pickle
from tqdm import tqdm
import numpy as np
from prompt import PROMPT
import json


def parse_args():

    parser = argparse.ArgumentParser(description="Test LLaDA model.")
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument(
        "--num_unmask_steps",
        type=int,
        default=64,
        help="Number of unmask steps to perform during prediction",
    )
    parser.add_argument("--p_uncond", type=float, default=0.1, help="Probability of unconditional training")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="Guidance scale")
    parser.add_argument(
        "--checkpoint_num",
        type=int,
        default=5925,
        help="Checkpoint number to load",
    )
    return parser.parse_args()


def test(args):
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    num_unmask_steps = args.num_unmask_steps
    cfg_scale = args.cfg_scale

    model = AutoModel.from_pretrained(
        "GSAI-ML/LLaDA-8B-Base",
        # quantization_config=BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        # ),
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Base", trust_remote_code=True
    )

    if args.checkpoint_num != 0:
        model = PeftModel.from_pretrained(
            model,
            os.path.join(
                "models",
                f"llada_{epochs}ep-{batch_size}bs-{lr}lr-{args.p_uncond}puncond",
                f"checkpoint-{args.checkpoint_num}",
            ),
        )
    model.eval()

    mask_id = tokenizer("<|mdm_mask|>", return_tensors="pt").input_ids[0][0].item()

    data_path = "paradetox_dataset"

    ds = load_from_disk(data_path)

    def preprocess(example):
        num_mask_tokens = 128
        hate_text = example["en_toxic_comment"]
        neutral_text = example["en_neutral_comment"]

        tokenized_input = tokenizer(
            PROMPT.format(hate_text=hate_text, neutral_text=neutral_text)
            + "<|endoftext|>",
            return_tensors="pt",
            truncation=True,
        )
        input_ids = tokenized_input.input_ids[0]
        attention_mask = tokenized_input.attention_mask[0]
        neutral_text_ids = tokenizer(
            neutral_text + "<|endoftext|>", return_tensors="pt", truncation=True
        ).input_ids[0]
        prompt_length = len(input_ids) - len(neutral_text_ids)

        labels = input_ids.clone()
        labels[:prompt_length] = -100  # Mask the input part for loss calculation

        masked_input_ids = input_ids.clone()
        masked_input_ids = torch.cat(
            (masked_input_ids[:prompt_length], torch.full((num_mask_tokens,), mask_id))
        )
        # masked_input_ids[prompt_length:][labels[prompt_length:] != -100] = mask_id

        example["input_ids"] = input_ids

        example["attention_mask"] = attention_mask
        example["labels"] = labels
        example["masked_input_ids"] = masked_input_ids
        example["prompt_length"] = prompt_length
        example["toxic_comment"] = hate_text

        return example

    ds_preprocessed = ds["test"].map(preprocess, num_proc=1)

    def predict(input_ids, attention_mask, num_steps, prompt_length):
        output_ids = input_ids.clone()
        num_masks = output_ids[output_ids == mask_id].numel()
        # print(f"Number of masks in the input: {num_masks}")
        steps = num_steps
        unmask_per_step = [num_masks // steps] * steps
        unmask_per_step = [
            unmask_per_step[i] + 1 if i < num_masks % steps else unmask_per_step[i]
            for i in range(steps)
        ]

        for unmask_num in tqdm(unmask_per_step):
            if unmask_num == 0:
                continue
            if cfg_scale != 1:
                uncond_output_ids = output_ids.clone()
                uncond_output_ids[:prompt_length] = mask_id
                output_ids_mix = torch.stack([output_ids, uncond_output_ids], dim=0)
                attention_mask_mix = torch.stack(
                    [attention_mask, attention_mask], dim=0
                )
                outputs = model(
                    input_ids=output_ids_mix,
                    attention_mask=attention_mask_mix,
                )
                logits = outputs.logits
                logits_cond = logits[0]
                logits_uncond = logits[1]
                logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)
            else:
                outputs = model(
                    input_ids=output_ids.unsqueeze(0),
                    attention_mask=attention_mask.unsqueeze(0),
                )
                logits = outputs.logits[0]
            pred_obj = torch.max(logits, dim=-1)
            pred_conf = pred_obj.values
            pred_ids = pred_obj.indices
            pred_conf[output_ids != mask_id] = -np.inf
            top_ids = torch.topk(pred_conf, unmask_num).indices
            output_ids[top_ids] = pred_ids[top_ids]

        return output_ids

    if args.checkpoint_num != 0:
        out_file = os.path.join(
            "results",
            f"llada_{epochs}ep-{batch_size}bs-{lr}lr-{args.p_uncond}puncond",
            f"result_{num_unmask_steps}un_{args.checkpoint_num}ch_{cfg_scale}cfgscale.jsonl",
        )
    else:
        out_file = os.path.join(
            "results",
            f"llada_untrained",
            f"result_{num_unmask_steps}un.jsonl",
        )

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    f_out = open(out_file, "w")

    preds = []
    actuals = []
    with torch.no_grad():
        for example in tqdm(ds_preprocessed):
            input_ids = torch.tensor(example["masked_input_ids"], device=model.device)
            attention_mask = torch.tensor(
                example["attention_mask"], device=model.device
            )
            output_ids = predict(input_ids, attention_mask, num_steps=num_unmask_steps, prompt_length=example["prompt_length"])

            pred = tokenizer.decode(output_ids[example["prompt_length"] :])
            actual = tokenizer.decode(example["input_ids"][example["prompt_length"] :])
            f_out.write(
                json.dumps(
                    {"hate": example["toxic_comment"], "pred": pred, "actual": actual}
                )
                + "\n"
            )
            f_out.flush()
            preds.append(pred)
            actuals.append(actual)

    f_out.close()


if __name__ == "__main__":
    args = parse_args()
    test(args)
