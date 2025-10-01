import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from typing import List, Tuple

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.2):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        embedded = self.dropout(self.embedding(x))

        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )

        outputs, (hidden, cell) = self.lstm(packed)

        # Unpack sequence
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        # Combine bidirectional hidden states
        hidden = hidden.view(self.num_layers, 2, x.size(0), self.hidden_size)
        hidden = torch.cat((hidden[:, 0, :, :], hidden[:, 1, :, :]), dim=2)

        cell = cell.view(self.num_layers, 2, x.size(0), self.hidden_size)
        cell = torch.cat((cell[:, 0, :, :], cell[:, 1, :, :]), dim=2)

        return outputs, (hidden, cell)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.2):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size + hidden_size * 2,
            hidden_size * 2,
            num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.out = nn.Linear(hidden_size * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, cell, encoder_outputs):
        embedded = self.dropout(self.embedding(input_token.unsqueeze(1)))

        # Get the top layer hidden state as query
        query = hidden[-1].unsqueeze(1)  # Shape: [batch_size, 1, hidden_size * 2]

        # Use scaled dot-product attention
        context = F.scaled_dot_product_attention(
            query=query, key=encoder_outputs, value=encoder_outputs
        )

        # Concatenate embedding and context
        lstm_input = torch.cat((embedded, context), dim=2)

        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        prediction = self.out(output.squeeze(1))

        return prediction, hidden, cell


class Seq2SeqModel(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        embed_size=128,
        hidden_size=256,
        num_layers=2,
        dropout=0.2,
    ):
        super(Seq2SeqModel, self).__init__()

        self.encoder = Encoder(
            src_vocab_size, embed_size, hidden_size, num_layers, dropout
        )
        self.decoder = Decoder(
            tgt_vocab_size, embed_size, hidden_size, num_layers, dropout
        )

    def forward(self, src, src_lens, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        max_len = tgt.size(1)
        vocab_size = self.decoder.vocab_size

        outputs = torch.zeros(batch_size, max_len, vocab_size).to(src.device)

        encoder_outputs, (hidden, cell) = self.encoder(src, src_lens)

        # First input to decoder is SOS token
        input_token = tgt[:, 0]

        for t in range(1, max_len):
            output, hidden, cell = self.decoder(
                input_token, hidden, cell, encoder_outputs
            )
            outputs[:, t, :] = output

            # Teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = tgt[:, t] if teacher_force else top1

        return outputs

    def generate_greedy(self, src, src_lens, max_len=50, sos_token=1, eos_token=2):
        """
        Generate sequences using greedy decoding (always pick the most likely token)
        """
        self.eval()
        with torch.no_grad():
            batch_size = src.size(0)
            device = src.device

            # Encode
            encoder_outputs, (hidden, cell) = self.encoder(src, src_lens)

            # Initialize with SOS token
            generated = torch.full(
                (batch_size, max_len), eos_token, dtype=torch.long, device=device
            )
            generated[:, 0] = sos_token

            input_token = torch.full(
                (batch_size,), sos_token, dtype=torch.long, device=device
            )

            for t in range(1, max_len):
                output, hidden, cell = self.decoder(
                    input_token, hidden, cell, encoder_outputs
                )

                # Greedy selection
                next_token = output.argmax(dim=1)
                generated[:, t] = next_token

                # Update input for next step
                input_token = next_token

                # Early stopping if all sequences have generated EOS
                if (next_token == eos_token).all():
                    break

            return generated

    def generate_beam_search(
        self,
        src,
        src_lens,
        beam_size=5,
        max_len=50,
        sos_token=1,
        eos_token=2,
        length_penalty=0.6,
    ):
        """
        Generate sequences using beam search
        Each beam is stored as: (score, sequence, hidden, cell)
        """
        self.eval()
        with torch.no_grad():
            device = src.device

            # Encode (single example)
            encoder_outputs, (hidden, cell) = self.encoder(src, src_lens)

            # Initialize beams: (score, sequence, hidden, cell)
            beams = [(0.0, [sos_token], hidden, cell)]
            completed = []

            for step in range(1, max_len):
                candidates = []

                for score, sequence, h, c in beams:
                    if sequence[-1] == eos_token:
                        # Add to completed sequences
                        length_norm = len(sequence) ** length_penalty
                        normalized_score = score / length_norm
                        completed.append((normalized_score, sequence))
                        continue

                    # Get next token probabilities
                    input_token = torch.tensor([sequence[-1]], device=device)
                    output, new_h, new_c = self.decoder(
                        input_token, h, c, encoder_outputs
                    )

                    # Get top k tokens
                    log_probs = F.log_softmax(output, dim=1)
                    top_k_probs, top_k_tokens = torch.topk(log_probs, beam_size, dim=1)

                    # Create candidates
                    for i in range(beam_size):
                        token = top_k_tokens[0, i].item()
                        prob = top_k_probs[0, i].item()
                        new_score = score + prob
                        new_sequence = sequence + [token]

                        candidates.append((new_score, new_sequence, new_h, new_c))

                # Keep top beam_size candidates
                candidates.sort(key=lambda x: x[0], reverse=True)
                beams = candidates[:beam_size]

                # Early stopping
                if len(completed) >= beam_size:
                    break

            # Add remaining beams to completed
            for score, sequence, _, _ in beams:
                if sequence[-1] != eos_token:
                    sequence = sequence + [eos_token]
                length_norm = len(sequence) ** length_penalty
                normalized_score = score / length_norm
                completed.append((normalized_score, sequence))

            # Return best sequence
            if completed:
                completed.sort(key=lambda x: x[0], reverse=True)
                best_sequence = completed[0][1]
                return torch.tensor(best_sequence, device=device), completed
            else:
                # Fallback to greedy
                print("Beam search failed, falling back to greedy decoding.")
                return (
                    self.generate_greedy(src, src_lens, max_len, sos_token, eos_token),
                    [],
                )

    def generate_batch_beam_search(
        self,
        src,
        src_lens,
        beam_size=5,
        max_len=50,
        sos_token=1,
        eos_token=2,
        length_penalty=0.6,
    ):
        """
        Generate sequences for a batch using beam search
        """
        batch_size = src.size(0)
        device = src.device
        generated = torch.full(
            (batch_size, max_len), eos_token, dtype=torch.long, device=device
        )

        for i in range(batch_size):
            # Process each example individually
            src_single = src[i : i + 1]
            src_len_single = src_lens[i : i + 1]

            best_sequence, _ = self.generate_beam_search(
                src_single,
                src_len_single,
                beam_size,
                max_len,
                sos_token,
                eos_token,
                length_penalty,
            )

            # Pad or truncate to max_len
            seq_len = min(len(best_sequence), max_len)
            generated[i, :seq_len] = best_sequence[:seq_len]

        return generated
