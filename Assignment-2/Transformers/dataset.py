from torch.utils.data import Dataset
import torch


class TransliterationDataset(Dataset):
    def __init__(self, data_pairs, src_vocab, tgt_vocab, max_len=50):
        self.data_pairs = data_pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        self.src_char2idx = {char: idx for idx, char in enumerate(src_vocab)}
        self.src_idx2char = {idx: char for char, idx in self.src_char2idx.items()}
        self.tgt_char2idx = {char: idx for idx, char in enumerate(tgt_vocab)}
        self.tgt_idx2char = {idx: char for char, idx in self.tgt_char2idx.items()}

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        src_word, tgt_word = self.data_pairs[idx]

        # Add special tokens
        src_seq = ["<SOS>"] + list(src_word.lower()) + ["<EOS>"]
        tgt_seq = ["<SOS>"] + list(tgt_word) + ["<EOS>"]

        # Convert to indices
        src_indices = [
            self.src_char2idx.get(char, self.src_char2idx["<UNK>"]) for char in src_seq
        ]
        tgt_indices = [
            self.tgt_char2idx.get(char, self.tgt_char2idx["<UNK>"]) for char in tgt_seq
        ]

        # Pad sequences
        src_indices = src_indices[: self.max_len]
        tgt_indices = tgt_indices[: self.max_len]

        src_len = len(src_indices)
        tgt_len = len(tgt_indices)

        # Pad to max_len
        src_indices += [self.src_char2idx["<PAD>"]] * (self.max_len - src_len)
        tgt_indices += [self.tgt_char2idx["<PAD>"]] * (self.max_len - tgt_len)

        return {
            "src": torch.tensor(src_indices, dtype=torch.long),
            "tgt": torch.tensor(tgt_indices, dtype=torch.long),
            "src_len": src_len,
            "tgt_len": tgt_len,
        }
