import math
import torch
import torch.nn as nn
import random
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        return x + self.pe[:, : x.size(1), :]


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_vocab_size,
        output_vocab_size,
        d_model=256,
        nhead=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=1024,
        dropout=0.1,
        max_len=5000,
        pad_token_id=0,  # <PAD>
        sos_token_id=1,  # <SOS>
        eos_token_id=2,  # <EOS>
        unk_token_id=3,  # <UNK>
    ):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        
        # Special token IDs (matching utils.py special_tokens order)
        self.pad_token_id = pad_token_id
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id
        self.unk_token_id = unk_token_id

        # Embedding layers
        self.input_embedding = nn.Embedding(input_vocab_size, d_model)
        self.output_embedding = nn.Embedding(output_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        # Transformer with batch_first=True
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # Changed to True
        )

        # Output projection
        self.output_projection = nn.Linear(d_model, output_vocab_size)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        initrange = 0.1
        self.input_embedding.weight.data.uniform_(-initrange, initrange)
        self.output_embedding.weight.data.uniform_(-initrange, initrange)
        self.output_projection.bias.data.zero_()
        self.output_projection.weight.data.uniform_(-initrange, initrange)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    def create_padding_mask(self, seq):
        """
        Create padding mask for sequences.
        Args:
            seq: Input sequence of shape (batch_size, seq_len)
        Returns:
            mask: Boolean mask of shape (batch_size, seq_len) where True indicates padding
        """
        # seq shape: (batch_size, seq_len)
        # mask shape: (batch_size, seq_len)
        mask = (seq == self.pad_token_id)
        return mask

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
        # Input shapes: src [batch_size, src_len], tgt [batch_size, tgt_len]
        
        # Create target input (shift right by one position)
        tgt_input = tgt[:, :-1]  # Remove last token
        tgt_output = tgt[:, 1:]  # Remove first token (expected output)

        # Create padding masks if not provided
        if src_key_padding_mask is None:
            src_key_padding_mask = self.create_padding_mask(src)
        
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = self.create_padding_mask(tgt_input)

        # Embeddings
        src_emb = self.input_embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
        tgt_emb = self.output_embedding(tgt_input) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))

        # Positional encoding
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_encoder(tgt_emb)

        # Apply dropout
        src_emb = self.dropout(src_emb)
        tgt_emb = self.dropout(tgt_emb)

        # Create causal mask for target
        tgt_mask = self.generate_square_subsequent_mask(tgt_input.size(1)).to(src.device)

        # Transformer forward pass with padding masks
        output = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        # Project to vocabulary
        output = self.output_projection(output)

        return output, tgt_output

    def encode(self, src, src_key_padding_mask=None):
        """
        Encode source sequence.
        """
        if src_key_padding_mask is None:
            src_key_padding_mask = self.create_padding_mask(src)

        src_emb = self.input_embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
        src_emb = self.pos_encoder(src_emb)
        src_emb = self.dropout(src_emb)

        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        return memory, src_key_padding_mask

    def decode_step(self, tgt, memory, tgt_mask=None, memory_key_padding_mask=None, tgt_key_padding_mask=None):
        """
        Single decoding step for inference.
        """
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = self.create_padding_mask(tgt)

        tgt_emb = self.output_embedding(tgt) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
        tgt_emb = self.pos_encoder(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)

        output = self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        output = self.output_projection(output)
        return output

    def generate_greedy(self, src, max_length=50):
        """
        Generate sequence using greedy decoding.
        """
        device = src.device
        batch_size = src.size(0)

        # Encode source
        memory, memory_key_padding_mask = self.encode(src)

        # Initialize target with SOS token
        tgt = torch.full((batch_size, 1), self.sos_token_id, dtype=torch.long, device=device)

        generated_tokens = []
        
        for _ in range(max_length):
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(device)
            
            output = self.decode_step(
                tgt,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

            # Get next token (last position of output)
            next_token_logits = output[:, -1, :]  # [batch_size, vocab_size]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # [batch_size, 1]

            # Add to generated tokens
            generated_tokens.append(next_token)
            tgt = torch.cat([tgt, next_token], dim=1)

            # Check if all sequences have generated EOS token
            if (next_token.squeeze(1) == self.eos_token_id).all():
                break

        # Concatenate all generated tokens (excluding SOS)
        if generated_tokens:
            return torch.cat(generated_tokens, dim=1)  # [batch_size, generated_len]
        else:
            return torch.empty((batch_size, 0), dtype=torch.long, device=device)

    def generate_batch_beam_search(self, src, beam_size=5, max_length=50):
        """
        Generate sequences using beam search for a batch.
        Note: This is a simplified version that processes each sequence independently.
        """
        batch_size = src.size(0)
        device = src.device
        
        results = []
        
        for i in range(batch_size):
            single_src = src[i:i+1]  # [1, seq_len]
            beam_result = self.generate_beam_search(single_src, beam_size, max_length)
            results.append(beam_result)
        
        # Pad results to same length
        max_len = max(r.size(1) for r in results) if results else 0
        if max_len == 0:
            return torch.empty((batch_size, 0), dtype=torch.long, device=device)
        
        padded_results = []
        for result in results:
            if result.size(1) < max_len:
                padding = torch.full((1, max_len - result.size(1)), self.pad_token_id, device=device)
                result = torch.cat([result, padding], dim=1)
            padded_results.append(result)
        
        return torch.cat(padded_results, dim=0)  # [batch_size, max_len]

    def generate_beam_search(self, src, beam_size=3, max_length=50):
        """
        Generate sequence using beam search for a single sequence.
        """
        device = src.device
        batch_size = src.size(0)
        
        if batch_size != 1:
            raise ValueError("Beam search currently supports batch_size=1 only")

        # Encode source
        memory, memory_key_padding_mask = self.encode(src)

        # Initialize beam with SOS token
        # beam_sequences: list of (sequence, score)
        beam_sequences = [(torch.tensor([[self.sos_token_id]], device=device), 0.0)]
        
        for _ in range(max_length):
            candidates = []

            for seq, score in beam_sequences:
                # If sequence ends with EOS, keep it as is
                if seq[0, -1] == self.eos_token_id:
                    candidates.append((seq, score))
                    continue
                
                tgt_mask = self.generate_square_subsequent_mask(seq.size(1)).to(device)
                
                output = self.decode_step(
                    seq,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                )
                
                # Get probabilities for next token
                next_token_probs = torch.log_softmax(output[0, -1, :], dim=-1)
                
                # Get top beam_size tokens
                top_probs, top_indices = next_token_probs.topk(beam_size)
                
                for prob, idx in zip(top_probs, top_indices):
                    new_seq = torch.cat([seq, idx.unsqueeze(0).unsqueeze(0)], dim=1)
                    new_score = score + prob.item()
                    candidates.append((new_seq, new_score))
            
            # Select top beam_size candidates
            candidates.sort(key=lambda x: x[1], reverse=True)
            beam_sequences = candidates[:beam_size]
            
            # Check if all beams end with EOS
            if all(seq[0, -1] == self.eos_token_id for seq, _ in beam_sequences):
                break
        
        # Return best sequence (excluding SOS token)
        best_seq, _ = beam_sequences[0]
        return best_seq[:, 1:]  # Remove SOS token

