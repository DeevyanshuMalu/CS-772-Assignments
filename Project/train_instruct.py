import os
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
import torch
from torch.nn import functional as F
import argparse
from prompt import PROMPT_INSTRUCT
import wandb


def parse_args():

    parser = argparse.ArgumentParser(description="Train LLaDA model.")
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--train_frac", type=float, default=0.95, help="Fraction of data to use for training")
    parser.add_argument("--p_uncond", type=float, default=0.1, help="Probability of unconditional training")
    parser.add_argument("--min_time", type=float, default=0.1, help="Minimum time for masking")
    parser.add_argument("--max_time", type=float, default=0.9, help="Maximum time for masking")
    return parser.parse_args()


def train(args):
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    train_frac = args.train_frac
    p_uncond = args.p_uncond
    min_time = args.min_time
    max_time = args.max_time

    assert 0.0 <= p_uncond <= 1.0, "p_uncond must be between 0 and 1"
    assert 0.0 <= min_time <= 1.0, "min_time must be between 0 and 1"
    assert 0.0 <= max_time <= 1.0, "max_time must be between 0 and 1"
    assert min_time < max_time, "min_time must be less than max_time"

    wandb.init(
        project="CS772_Project",
        config={"epochs": epochs, "batch_size": batch_size, "lr": lr, "p_uncond": p_uncond},
    )

    model = AutoModel.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct",
        # quantization_config=BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        # ),
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    mask_id = tokenizer("<|mdm_mask|>", return_tensors="pt").input_ids[0][0].item()

    data_path = "paradetox_dataset"

    ds = load_from_disk(data_path)

    def preprocess(example):
        hate_text = example["en_toxic_comment"]
        neutral_text = example["en_neutral_comment"]

        input_ids = tokenizer(
            PROMPT_INSTRUCT.format(hate_text=hate_text, neutral_text=neutral_text)
            + "<|endoftext|>",
            return_tensors="pt",
            truncation=True,
        ).input_ids[0]
        neutral_text_ids = tokenizer(
            neutral_text + "\n<|endoftext|>", return_tensors="pt", truncation=True
        ).input_ids[0]
        prompt_length = len(input_ids) - len(neutral_text_ids)

        labels = input_ids.clone()
        labels[:prompt_length] = -100  # Mask the input part for loss calculation

        # t = torch.rand(1).item()

        # mask = (torch.rand(input_ids[prompt_length:].shape, dtype=torch.float) < t) & (
        #     labels[prompt_length:] != -100
        # )
        # input_ids[prompt_length:][mask] = mask_id
        # labels[prompt_length:][~mask] = -100

        example["input_ids"] = input_ids
        example["labels"] = labels
        # example["t"] = t

        example["prompt_length"] = prompt_length

        return example

    def collator(batch):
        input_ids = [example["input_ids"] for example in batch]
        labels = [example["labels"] for example in batch]
        ts = torch.rand(len(batch)).tolist()
        ts = [min_time + (max_time - min_time) * t for t in ts]
        uncond = torch.rand(len(batch)).tolist()
        prompt_lengths = [example["prompt_length"] for example in batch]
        # inputs = [example['input'] for example in batch]
        # outputs = [example['output'] for example in batch]

        for i in range(len(batch)):
            prompt_length = prompt_lengths[i]
            t = ts[i]
            i_ids = input_ids[i]
            i_ids = torch.tensor(i_ids)
            l_ids = labels[i]
            l_ids = torch.tensor(l_ids)
            mask = (torch.rand(i_ids[prompt_length:].shape, dtype=torch.float) < t) & (
                l_ids[prompt_length:] != -100
            )
            if uncond[i] < p_uncond:
                i_ids[:prompt_length] = mask_id
            i_ids[prompt_length:][mask] = mask_id
            l_ids[prompt_length:][~mask] = -100
            input_ids[i] = i_ids.tolist()
            labels[i] = l_ids.tolist()

        max_length = 256
        padded_input_ids = torch.tensor(
            [
                ids + [tokenizer.pad_token_id] * (max_length - len(ids))
                for ids in input_ids
            ]
        )
        padded_labels = torch.tensor(
            [lbl + [-100] * (max_length - len(lbl)) for lbl in labels]
        )
        attention_mask = torch.tensor(
            [[1] * len(ids) + [0] * (max_length - len(ids)) for ids in input_ids]
        )

        return {
            "input_ids": padded_input_ids,
            "labels": padded_labels,
            "prompt_lengths": torch.tensor(prompt_lengths),
            "attention_mask": attention_mask,
            "t": torch.tensor(ts),
            # "inputs": inputs,
            # "outputs": outputs,
        }

    ds_preprocessed = ds["train"].map(preprocess, num_proc=1)
    ds_preprocessed = ds_preprocessed.shuffle(seed=42)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=8,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj"],
    )

    model = get_peft_model(model, lora_config)
    model = model.to("cuda")
    model.print_trainable_parameters()

    class dLLMTrainer(Trainer):
        def compute_loss(
            self, model, inputs, num_items_in_batch=None, return_outputs=False
        ):
            labels, t, num_prompt_tokens = (
                inputs.pop("labels"),
                inputs.pop("t"),
                inputs.pop("prompt_lengths"),
            )
            outputs = model(**inputs)
            logits = outputs.logits

            unscaled_loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]), labels.view(-1), reduction="none"
            ).view(logits.shape[0], -1)
            loss = unscaled_loss / t.reshape(-1, 1)
            loss = loss.sum() / (inputs["input_ids"].numel() - num_prompt_tokens.sum())
            return loss if not return_outputs else (loss, outputs)

    training_args = TrainingArguments(
        output_dir=os.path.join(
            "models_instruct",
            f"llada_{epochs}ep-{batch_size}bs-{lr}lr-{p_uncond}puncond",
        ),
        overwrite_output_dir=True,
        fp16=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        num_train_epochs=epochs,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        # save_total_limit=1,
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_loss",
        # greater_is_better=False,
        report_to=["wandb"],
        label_names=["labels"],
        remove_unused_columns=False,
    )

    trainer = dLLMTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=ds_preprocessed.select(range(int(train_frac * len(ds_preprocessed)))),
        eval_dataset=ds_preprocessed.select(
            range(int(train_frac * len(ds_preprocessed)), len(ds_preprocessed))
        ),
        data_collator=collator,
    )

    trainer.train()


if __name__ == "__main__":
    args = parse_args()
    train(args)
