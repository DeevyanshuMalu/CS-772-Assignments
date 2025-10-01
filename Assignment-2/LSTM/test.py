import torch
from torch.utils.data import DataLoader
import numpy as np
import pickle
import os
import random
import argparse
from tqdm import tqdm
from models import *
from dataset import *
from utils import *

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)
random.seed(42)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LSTM Seq2Seq model for English-Hindi transliteration"
    )

    # Data paths
    parser.add_argument(
        "--test_data",
        type=str,
        default="../hin/hin_test.jsonl",
        help="Path to test data file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models",
        help="Directory to save model checkpoints",
    )

    # Model hyperparameters
    parser.add_argument(
        "--embed_size", type=int, default=128, help="Embedding dimension"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=256, help="Hidden dimension of LSTM"
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of LSTM layers"
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument(
        "--max_len", type=int, default=50, help="Maximum sequence length"
    )

    # Training hyperparameters
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="Learning rate"
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--teacher_forcing_ratio",
        type=float,
        default=0.5,
        help="Teacher forcing ratio during training",
    )
    parser.add_argument(
        "--min_freq", type=int, default=1, help="Minimum frequency for vocabulary"
    )

    # Testing options
    parser.add_argument(
        "--epoch_to_test",
        type=int,
        default=25,
        help="Epoch number of the model to test",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Device to use: "auto", "cpu", or "cuda"',
    )

    return parser.parse_args()


def test(args):
    model_dir = os.path.join(
        args.output_dir,
        f"{args.learning_rate}lr_{args.num_epochs}epochs_{args.teacher_forcing_ratio}tf",
    )

    # Load data
    print("Loading data...")
    test_pairs = load_data(args.test_data)

    print(f"Test samples: {len(test_pairs)}")

    vocab_path = os.path.join(args.output_dir, "vocab.pkl")
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    src_vocab = vocab["src_vocab"]
    tgt_vocab = vocab["tgt_vocab"]

    test_dataset = TransliterationDataset(
        test_pairs, src_vocab, tgt_vocab, args.max_len
    )

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    model = Seq2SeqModel(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    model_path = os.path.join(model_dir, f"checkpoint_epoch_{args.epoch_to_test}.pt")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict["model_state_dict"])
    model.eval()
    preds_greedy = []
    preds_beam = []
    actuals = []

    for batch in tqdm(test_loader, desc="Testing", total=len(test_loader)):
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        src_lens = batch["src_len"]

        with torch.no_grad():
            output_tokens_greedy = model.generate_greedy(src, src_lens)
            output_tokens_beam = model.generate_batch_beam_search(
                src, src_lens, beam_size=5
            )

        src_idx2char = test_dataset.src_idx2char
        tgt_idx2char = test_dataset.tgt_idx2char

        for i in range(src.size(0)):
            src_seq = "".join(
                [
                    src_idx2char[idx.item()]
                    for idx in src[i]
                    if src_idx2char[idx.item()] not in ["<PAD>", "<SOS>", "<EOS>"]
                ]
            )
            tgt_seq = "".join(
                [
                    tgt_idx2char[idx.item()]
                    for idx in tgt[i]
                    if tgt_idx2char[idx.item()] not in ["<PAD>", "<SOS>", "<EOS>"]
                ]
            )
            pred_seq_greedy = "".join(
                [
                    tgt_idx2char[idx.item()]
                    for idx in output_tokens_greedy[i]
                    if tgt_idx2char[idx.item()] not in ["<PAD>", "<SOS>", "<EOS>"]
                ]
            )
            pred_seq_beam = "".join(
                [
                    tgt_idx2char[idx.item()]
                    for idx in output_tokens_beam[i]
                    if tgt_idx2char[idx.item()] not in ["<PAD>", "<SOS>", "<EOS>"]
                ]
            )

            preds_greedy.append(pred_seq_greedy)
            preds_beam.append(pred_seq_beam)
            actuals.append(tgt_seq)

            # print(f"Source: {src_seq}")
            # print(f"Target: {tgt_seq}")
            # print(f"Predicted (Greedy): {pred_seq_greedy}")
            # print(f"Predicted (Beam Search): {pred_seq_beam}")
            # print("-" * 50)
        # break

    accuracy_greedy = compute_accuracy(actuals, preds_greedy)
    accuracy_beam = compute_accuracy(actuals, preds_beam)
    f1_greedy = get_f1_score(actuals, preds_greedy)
    f1_beam = get_f1_score(actuals, preds_beam)

    with open(os.path.join(model_dir, f"test_results_{args.epoch_to_test}epochs.txt"), "w") as f:
        f.write(f"Greedy Decoding Accuracy: {accuracy_greedy:.4f}\n")
        f.write(f"Beam Search Decoding Accuracy: {accuracy_beam:.4f}\n")
        f.write(f"Greedy Decoding F1 Score: {f1_greedy:.4f}\n")
        f.write(f"Beam Search Decoding F1 Score: {f1_beam:.4f}\n")


if __name__ == "__main__":
    args = parse_args()
    test(args)
