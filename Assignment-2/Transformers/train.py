import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pickle
import os
import random
import argparse
import wandb
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
        description="Train Transformer Seq2Seq model for English-Hindi transliteration"
    )

    # Data paths
    parser.add_argument(
        "--train_data",
        type=str,
        default="../hin/hin_train_chosen.jsonl",
        help="Path to training data file",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default="../hin/hin_valid.jsonl",
        help="Path to validation data file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models",
        help="Directory to save model checkpoints",
    )

    # Model hyperparameters
    parser.add_argument(
        "--d_model", type=int, default=256, help="Model dimension (embedding size)"
    )
    parser.add_argument(
        "--nhead", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--num_encoder_layers", type=int, default=2, help="Number of encoder layers"
    )
    parser.add_argument(
        "--num_decoder_layers", type=int, default=2, help="Number of decoder layers"
    )
    parser.add_argument(
        "--dim_feedforward", type=int, default=1024, help="Feedforward network dimension"
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
        "--min_freq", type=int, default=1, help="Minimum frequency for vocabulary"
    )

    # Training options
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use: cpu, cuda, or auto"
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="Patience for learning rate scheduler"
    )
    parser.add_argument(
        "--lr_factor", type=float, default=0.8, help="Factor to reduce learning rate"
    )
    parser.add_argument(
        "--grad_clip", type=float, default=1.0, help="Gradient clipping threshold"
    )
    parser.add_argument(
        "--save_freq", type=int, default=5, help="Save model every N epochs"
    )

    # Wandb options
    parser.add_argument(
        "--use_wandb", action="store_true", help="Use Weights & Biases for logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="transliteration-transformer",
        help="Wandb project name",
    )

    return parser.parse_args()


def train_model(args):
    # Initialize wandb if specified
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"{args.learning_rate}lr_{args.num_epochs}epochs",
            config=vars(args)
        )

    # Create output directory
    model_dir = os.path.join(
        args.output_dir,
        f"{args.learning_rate}lr_{args.num_epochs}epochs",
    )
    os.makedirs(model_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    train_pairs = load_data(args.train_data)
    val_pairs = load_data(args.val_data)

    print(f"Training samples: {len(train_pairs)}")
    print(f"Validation samples: {len(val_pairs)}")

    # Build vocabulary
    src_vocab, tgt_vocab = build_vocab(train_pairs, args.min_freq)

    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")

    # Save vocabulary
    vocab_path = os.path.join(args.output_dir, "vocab.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump({"src_vocab": src_vocab, "tgt_vocab": tgt_vocab}, f)
    print(f"Vocabulary saved to {vocab_path}")

    # Create datasets
    train_dataset = TransliterationDataset(
        train_pairs, src_vocab, tgt_vocab, args.max_len
    )
    val_dataset = TransliterationDataset(val_pairs, src_vocab, tgt_vocab, args.max_len)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Initialize Transformer model
    model = TransformerModel(
        input_vocab_size=len(src_vocab),  # Use source vocab size
        output_vocab_size=len(tgt_vocab),  # Use target vocab size
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_len=args.max_len,
        pad_token_id=train_dataset.tgt_char2idx["<PAD>"],  # <PAD>
        sos_token_id=train_dataset.tgt_char2idx["<SOS>"],  # <SOS>
        eos_token_id=train_dataset.tgt_char2idx["<EOS>"],  # <EOS>
        unk_token_id=train_dataset.tgt_char2idx["<UNK>"],  # <UNK>
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.tgt_char2idx["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=args.patience, factor=args.lr_factor
    )

    # Resume from checkpoint if specified
    start_epoch = 0

    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        # Training
        model.train()
        train_loss = 0

        for batch_idx, batch in tqdm(
            enumerate(train_loader),
            desc=f"Epoch {epoch+1}/{args.num_epochs} - Training",
            total=len(train_loader),
        ):
            src = batch["src"].to(device)  # [batch_size, seq_len]
            tgt = batch["tgt"].to(device)  # [batch_size, seq_len]

            optimizer.zero_grad()

            # Forward pass - no need to transpose with batch_first=True
            output, target = model(src, tgt)

            # Reshape for loss calculation
            output = output.contiguous().view(-1, len(tgt_vocab))
            target = target.contiguous().view(-1)

            loss = criterion(output, target)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", total=len(val_loader)):
                src = batch["src"].to(device)
                tgt = batch["tgt"].to(device)

                # Forward pass
                output, target = model(src, tgt)

                # Reshape for loss calculation
                output = output.contiguous().view(-1, len(tgt_vocab))
                target = target.contiguous().view(-1)

                loss = criterion(output, target)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{args.num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Log epoch-level metrics to wandb
        if args.use_wandb:
            wandb.log(
                {
                    "epoch/train_loss": train_loss,
                    "epoch/val_loss": val_loss,
                    "epoch/learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch/epoch": epoch + 1,
                }
            )

        # Save checkpoint periodically
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(model_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "src_vocab": src_vocab,
                    "tgt_vocab": tgt_vocab,
                    "config": vars(args),
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint to {checkpoint_path}")

        print("-" * 50)

    print("Training completed!")


if __name__ == "__main__":
    args = parse_args()
    train_model(args)
