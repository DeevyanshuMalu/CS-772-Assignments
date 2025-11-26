import argparse
import json
import os
from box import Box

import torch
import numpy as np

def build_paths(ckpt_num: int, cfg_scale: float, svdd: bool, baseline: bool):
    # fixed params from notebook
    epochs = 3
    lr = 2e-4
    batch_size = 8
    num_unmask_steps = 128
    p_uncond = 0.0
    num_cands = 4
    instruct = True

    instruct_str = '-instruct' if instruct else ''
    svdd_str = '-svdd' if svdd else ''
    num_cands_str = f'_{num_cands}numcands' if svdd else ''

    if baseline:
        results_path = f'./results{instruct_str}/baseline.jsonl'
        output_path = f'./results{instruct_str}/baseline_eval.json'
    else:
        if ckpt_num == 0:
            results_path = f'./results{instruct_str}{svdd_str}/llada_untrained/result_{num_unmask_steps}un_{cfg_scale}cfgscale{num_cands_str}.jsonl'
            output_path = f'./results{instruct_str}{svdd_str}/llada_untrained/eval_{num_unmask_steps}un_{cfg_scale}cfgscale{num_cands_str}.json'
        else:
            results_path = f'./results{instruct_str}{svdd_str}/llada_{epochs}ep-{batch_size}bs-{lr}lr-{p_uncond}puncond/result_{num_unmask_steps}un_{ckpt_num}ch_{cfg_scale}cfgscale{num_cands_str}.jsonl'
            output_path = f'./results{instruct_str}{svdd_str}/llada_{epochs}ep-{batch_size}bs-{lr}lr-{p_uncond}puncond/eval_{num_unmask_steps}un_{ckpt_num}ch_{cfg_scale}cfgscale{num_cands_str}.json'
    return results_path, output_path

def load_preds(results_path):
    preds, actuals, hates = [], [], []
    with open(results_path, 'r') as f:
        for line in f:
            result = json.loads(line)
            pred = result.get('pred', '')
            pred = pred.split('<|endoftext|>')[0].strip()
            pred = pred.split('\n')[0]
            # pred = pred.split('\n')[-1]
            preds.append(pred.lower())

            actual = result.get('actual', '')
            actual = actual.split('<|endoftext|>')[0].strip()
            actuals.append(actual.lower())

            hate = result.get('hate', '')
            hates.append(hate.lower())
    return preds, actuals, hates

def main():
    parser = argparse.ArgumentParser(description='Evaluate results jsonl for checkpoints')
    parser.add_argument('--ckpt_num', type=int, required=True, help='Checkpoint number (0 for untrained)')
    parser.add_argument('--cfg_scale', type=float, required=True, help='CFG scale used during generation')
    parser.add_argument('--svdd', action='store_true', help='Use SVDD variant (flag)')
    parser.add_argument('--baseline', action='store_true', help='Evaluate baseline (flag)')
    args_cli = parser.parse_args()

    results_path, output_path = build_paths(args_cli.ckpt_num, args_cli.cfg_scale, args_cli.svdd, args_cli.baseline)
    print("Results will be read from:", results_path)
    print("Results will be saved to:", output_path)

    if not os.path.exists(results_path):
        raise FileNotFoundError(f"{results_path} not found")

    preds, actuals, hates = load_preds(results_path)
    print(f"Loaded {len(preds)} examples")

    # rebuild notebook args Box
    args = Box({
        'batch_size': 32,
        'cola_classifier_path': '/content/drive/MyDrive/style_transfer/cola_classifier',
        'wieting_tokenizer_path': 'sim.sp.30k.model',
        'wieting_model_path': 'sim.pt',
        't1': 75., 't2': 70., 't3': 12.
    })

    # import metric functions (may require installed paradetox package)
    try:
        from paradetox.evaluation_detox.metric_tools.style_transfer_accuracy import classify_preds
        from paradetox.evaluation_detox.metric_tools.content_similarity import flair_sim
        from paradetox.evaluation_detox.metric_tools.fluency import cola_fluency
        from paradetox.evaluation_detox.metric_tools.joint_metrics import get_j
    except Exception as e:
        raise ImportError("Required paradetox metric tools not available: " + str(e))

    # Style Transfer Accuracy
    accuracy_by_sent = classify_preds(args, preds)
    accuracy = float(np.mean(accuracy_by_sent))

    # Content similarity
    emb_sim_stats = flair_sim(args, hates, preds)
    emb_sim = float(emb_sim_stats.mean())

    # Fluency
    cola_stats = cola_fluency(preds)
    cola_acc = float(sum(cola_stats) / len(preds))
    cola_stats_tensor = torch.tensor(cola_stats)

    # Fluency similarity
    cola_stats_hates = cola_fluency(hates)
    cola_stats_hates_tensor = torch.tensor(cola_stats_hates)
    cola_sim_accuracy = float(torch.sum(cola_stats_tensor == cola_stats_hates_tensor).item() / cola_stats_tensor.shape[0])

    # Joint metrics
    # Ensure inputs to get_j are plain numpy arrays / Python floats (no mixed torch/numpy/list types)
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.array(x)

    accuracy_arr = to_numpy(accuracy_by_sent).astype(float)
    # emb_sim_stats may be an array-like of per-sentence similarities
    sim_arr = to_numpy(emb_sim_stats).astype(float)
    cola_arr = to_numpy(cola_stats).astype(float)

    # get_j expects array-like elementwise multiplications; pass numpy arrays
    joint = float(get_j(args, accuracy_arr, sim_arr, cola_arr, preds))
    # For the variant that uses a single float for FL acc, pass the float (no array)
    joint_with_fl_acc = float(get_j(args, accuracy_arr, sim_arr, float(cola_sim_accuracy), preds))

    eval_dict = {
        "STA": accuracy,
        "SIM": emb_sim,
        "FL": cola_acc,
        "FL_acc": cola_sim_accuracy,
        "J": joint,
        "J_with_FL_acc": joint_with_fl_acc
    }

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(eval_dict, f, indent=4)

    print("Saved eval:", eval_dict)

if __name__ == "__main__":
    main()