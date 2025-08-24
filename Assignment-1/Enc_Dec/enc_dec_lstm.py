from torch import nn
import torch
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F
from typing import Optional, Tuple

random.seed(42)
torch.manual_seed(42)

device = torch.device("xpu" if torch.xpu.is_available() else "cpu")

class MyLSTM(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 bias: bool = True,
                 batch_first: bool = False,
                 dropout: float = 0.0,
                 bidirectional: bool = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.num_directions = 2 if bidirectional else 1

        # create parameters with the same names as nn.LSTM for compatibility / weight-copying
        for layer in range(num_layers):
            for direction in range(self.num_directions):
                suffix = f"_reverse" if direction == 1 else ""
                layer_index = f"_l{layer}"
                suffix_name = f"{layer_index}{suffix}"

                ih_name = f"weight_ih{suffix_name}"
                hh_name = f"weight_hh{suffix_name}"
                # Fix: input_size for first layer, hidden_size * num_directions for others
                if layer == 0:
                    ih_dim = input_size
                else:
                    ih_dim = hidden_size * self.num_directions
                self.register_parameter(ih_name, nn.Parameter(torch.Tensor(4 * hidden_size, ih_dim)))
                self.register_parameter(hh_name, nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size)))

                if bias:
                    b_ih_name = f"bias_ih{suffix_name}"
                    b_hh_name = f"bias_hh{suffix_name}"
                    self.register_parameter(b_ih_name, nn.Parameter(torch.Tensor(4 * hidden_size)))
                    self.register_parameter(b_hh_name, nn.Parameter(torch.Tensor(4 * hidden_size)))
                else:
                    setattr(self, f"bias_ih{suffix_name}", None)
                    setattr(self, f"bias_hh{suffix_name}", None)

        # dropout layer used between layers (not between time steps)
        self.dropout_layer = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Follow default PyTorch LSTM init: uniform with bound based on hidden_size
        stdv = 1.0 / (self.hidden_size ** 0.5)
        for name, param in self.named_parameters():
            if param is None:
                continue
            nn.init.uniform_(param, -stdv, stdv)

    def _get_param(self, name: str) -> Optional[torch.Tensor]:
        return getattr(self, name, None)

    def forward(self, input: torch.Tensor,
                hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        if self.batch_first:
            input = input.transpose(0, 1)  # (seq_len, batch, input_size)

        seq_len, batch_size, input_size = input.shape
        assert input_size == self.input_size

        if hx is None:
            h_t = input.new_zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)
            c_t = input.new_zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)
        else:
            h_t, c_t = hx

        layer_input = input

        final_h = []
        final_c = []

        for layer in range(self.num_layers):
            outputs_from_directions = []

            for direction in range(self.num_directions):
                reverse = (direction == 1)
                suffix = f"_reverse" if reverse else ""
                layer_index = f"_l{layer}"
                suffix_name = f"{layer_index}{suffix}"

                w_ih = self._get_param(f"weight_ih{suffix_name}")
                w_hh = self._get_param(f"weight_hh{suffix_name}")
                b_ih = self._get_param(f"bias_ih{suffix_name}") if self.bias else None
                b_hh = self._get_param(f"bias_hh{suffix_name}") if self.bias else None

                idx = layer * self.num_directions + direction
                h_prev = h_t[idx]
                c_prev = c_t[idx]

                if reverse:
                    layer_in = torch.flip(layer_input, [0])
                else:
                    layer_in = layer_input

                # gates: (seq_len, batch, 4*hidden)
                gates = F.linear(layer_in, w_ih, b_ih) + F.linear(h_prev.unsqueeze(0).expand(seq_len, -1, -1), w_hh, b_hh)
                i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=2)
                i_gate = torch.sigmoid(i_gate)
                f_gate = torch.sigmoid(f_gate)
                g_gate = torch.tanh(g_gate)
                o_gate = torch.sigmoid(o_gate)

                h_cur = h_prev
                c_cur = c_prev
                outputs_time = []

                for t in range(seq_len):
                    c_cur = f_gate[t] * c_cur + i_gate[t] * g_gate[t]
                    h_cur = o_gate[t] * torch.tanh(c_cur)
                    outputs_time.append(h_cur.unsqueeze(0))

                outputs_time = torch.cat(outputs_time, dim=0)
                if reverse:
                    outputs_time = torch.flip(outputs_time, [0])

                outputs_from_directions.append(outputs_time)
                final_h.append(h_cur.unsqueeze(0))
                final_c.append(c_cur.unsqueeze(0))

            if self.num_directions == 1:
                layer_output = outputs_from_directions[0]
            else:
                layer_output = torch.cat(outputs_from_directions, dim=2)

            # Update layer_input for next layer
            if (self.dropout > 0.0) and (layer < self.num_layers - 1):
                layer_input = self.dropout_layer(layer_output)
            else:
                layer_input = layer_output

        h_n = torch.cat(final_h, dim=0)
        c_n = torch.cat(final_c, dim=0)
        output = layer_input

        if self.batch_first:
            output = output.transpose(0, 1)
        # print(output.shape, (h_n.shape, c_n.shape), len(final_h), len(final_c), final_h[0].shape, final_c[0].shape)
        return output, (h_n, c_n)

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, embedding_matrix):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight.data.copy_(embedding_matrix)
        self.embedding.weight.requires_grad = False  # Freeze embeddings
        self.lstm = MyLSTM(embed_dim, hidden_dim, batch_first=True, num_layers=3)
    
    def forward(self, input, lengths):
        embedded = self.embedding(input)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, tag_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(tag_size, embed_dim, padding_idx=0)
        self.lstm = MyLSTM(embed_dim, hidden_dim, batch_first=True, num_layers=3)
        self.fc = nn.Linear(hidden_dim, tag_size)

    def forward(self, input, hidden, cell):
        embedded = self.embedding(input)
        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        predictions = self.fc(outputs)
        return predictions, hidden, cell
    
class Encoder_Decoder_Model(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, tag_size, sos_index, embedding_matrix):
        super().__init__()
        self.encoder = Encoder(vocab_size, embed_dim, hidden_dim, embedding_matrix)
        self.decoder = Decoder(tag_size, embed_dim, hidden_dim)
        self.sos_index = sos_index
        self.device = device

    def forward(self, input_seq, input_tags, input_lengths):
        hidden, cell = self.encoder(input_seq, input_lengths)
        input_tags = torch.cat([torch.tensor([[self.sos_index]]*len(input_seq), device=self.device), input_tags[:, :-1]], dim=1)
        output, hidden, cell = self.decoder(input_tags, hidden, cell)
        return output, hidden, cell

    def generate(self, input_seq, max_len, lengths):
        with torch.no_grad():
            hidden, cell = self.encoder(input_seq, lengths)
            input_tag = torch.tensor([[self.sos_index]]*len(input_seq), device=self.device)
            for _ in range(max_len):
                output = self.decoder(input_tag, hidden, cell)
                top1 = output[0].argmax(dim=-1)
                input_tag = torch.cat([input_tag, top1[:, -1].unsqueeze(1)], dim=1)
            return input_tag
