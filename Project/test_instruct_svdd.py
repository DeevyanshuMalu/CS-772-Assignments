import argparse
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import os
import json

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util

from prompt import PROMPT_INSTRUCT

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@ torch.no_grad()
def generate(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False, sentence_model=None, tokenizer=None, num_candidates=4, hates=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        logits_eos_inf: Whether to set the logits of EOS token to -inf. See Appendix B.4 of LLaDA for details
        confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf. See Appendix B.4 of LLaDA for details
        sentence_model: Sentence transformer model for re-ranking candidates.
        num_candidates: Number of candidates for re-ranking.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    hates_emb = sentence_model.encode(hates, convert_to_tensor=True, normalize_embeddings=True)

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in tqdm(range(steps)):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
            
            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf
            
            candidates = []
            for _ in range(num_candidates):
                if remasking == 'low_confidence':
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x_copy = x.clone()
                x_copy[transfer_index] = x0[transfer_index]
                candidates.append(x_copy)
                # x[transfer_index] = x0[transfer_index]
            candidates = torch.stack(candidates, dim=1)  # b, num_candidates, l

            if i == steps - 1:
                x = candidates[:, 0]  # Select the first candidate if it's the last step
                continue

            all_scores = []
            for c in range(candidates.shape[1]):
                cands = candidates[:, c]  # b, l
                mask_index = (cands == mask_id)
                num_masks = mask_index.sum(dim=1)
                # if cfg_scale > 0.:
                #     un_cands = cands.clone()
                #     un_cands[prompt_index] = mask_id
                #     cands_ = torch.cat([cands, un_cands], dim=0)
                #     if attention_mask is not None:
                #         attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                #     logits = model(cands_, attention_mask=attention_mask_).logits
                #     logits, un_logits = torch.chunk(logits, 2, dim=0)
                #     logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                # else:
                #     logits = model(cands, attention_mask=attention_mask).logits
                logits = model(cands, attention_mask=attention_mask).logits
                

                if logits_eos_inf:
                    logits[:, :, 126081] = -torch.inf

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                cands0 = torch.argmax(logits_with_noise, dim=-1) # b, l
                
                if confidence_eos_eot_inf:
                    logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

                cands0 = torch.where(mask_index, cands0, cands)

                transfer_index = torch.zeros_like(cands0, dtype=torch.bool, device=cands0.device)
                for j in range(cands0.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_masks[j])
                    transfer_index[j, select_index] = True

                cands_copy = cands.clone()
                cands_copy[transfer_index] = cands0[transfer_index]

                cands_text = tokenizer.batch_decode(cands_copy[:, prompt.shape[1]:], skip_special_tokens=True) # b of strings
                
                cands_emb = sentence_model.encode(cands_text, convert_to_tensor=True, normalize_embeddings=True)

                scores = torch.diagonal(util.cos_sim(cands_emb, hates_emb))  # b
                all_scores.append(scores)
            all_scores = torch.stack(all_scores, dim=1)  # b, num_candidates
            best_indices = torch.argmax(all_scores, dim=1)  # b
            for b in range(x.shape[0]):
                x[b] = candidates[b, best_indices[b]]

    return x

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
    parser.add_argument("--gen_length", type=int, default=128, help="Generation length")
    parser.add_argument("--num_candidates", type=int, default=4, help="Number of candidates for re-ranking")
    parser.add_argument("--p_uncond", type=float, default=0.0, help="Probability of unconditional training")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="Guidance scale")
    parser.add_argument(
        "--checkpoint_num",
        type=int,
        default=2321,
        help="Checkpoint number to load",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    num_unmask_steps = args.num_unmask_steps
    gen_length = args.gen_length
    num_candidates = args.num_candidates
    cfg_scale = args.cfg_scale
    p_uncond = args.p_uncond
    checkpoint_num = args.checkpoint_num
    test_batch_size = 8

    device = 'cuda'

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

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

    # The LLaDA architecture theoretically supports both left-padding and right-padding. 
    # However, the sampling code implementation is simpler with left-padding.
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'

    # If the padding ID equals the mask ID, you need to modify our generate function to achieve correct inference.
    assert tokenizer.pad_token_id != 126336

    data_path = "paradetox_dataset"
    ds = load_from_disk(data_path)
    ds_test = ds["test"]

    prompts = [PROMPT_INSTRUCT.format(hate_text=example["en_toxic_comment"], neutral_text="")[:-1] for example in ds_test]

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    # messages = [{"role": "user", "content": prompt} for prompt in prompts]
    # prompts = [tokenizer.apply_chat_template([message], add_generation_prompt=True, tokenize=False) for message in messages]

    encoded_outputs = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt"
    )
    input_ids = encoded_outputs['input_ids'].to(device)
    attention_mask = encoded_outputs['attention_mask'].to(device)

    print(input_ids.shape[0])

    if checkpoint_num != 0:
        out_file = os.path.join(
            "results-instruct-svdd",
            f"llada_{epochs}ep-{batch_size}bs-{lr}lr-{p_uncond}puncond",
            f"result_{num_unmask_steps}un_{checkpoint_num}ch_{cfg_scale}cfgscale_{num_candidates}numcands.jsonl",
        )
    else:
        out_file = os.path.join(
            "results-instruct-svdd",
            f"llada_untrained",
            f"result_{num_unmask_steps}un_{cfg_scale}cfgscale_{num_candidates}numcands.jsonl",
        )

    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    print(f"Writing results to {out_file}")

    with open(out_file, 'w') as f_out:
        for i in tqdm(range(0, input_ids.shape[0], test_batch_size)):
            hates = [example["en_toxic_comment"] for example in ds_test.select(range(i, min(i + test_batch_size, input_ids.shape[0])))]
            out = generate(model, input_ids[i:i + test_batch_size], attention_mask=attention_mask[i:i + test_batch_size], steps=num_unmask_steps, gen_length=gen_length, block_length=gen_length, temperature=0., cfg_scale=cfg_scale-1, remasking='random', sentence_model=sentence_model, tokenizer=tokenizer, num_candidates=num_candidates, hates=hates)
            output = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
            ds_rows = ds_test.select(range(i, min(i + test_batch_size, input_ids.shape[0])))
            for o, example in zip(output, ds_rows):
                f_out.write(json.dumps(
                    {"hate": example["en_toxic_comment"], "pred": o, "actual": example["en_neutral_comment"]}
                ) + "\n")
                f_out.flush()


if __name__ == '__main__':
    main()