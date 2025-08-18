from torch import nn
import torch
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

random.seed(42)

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
    
    def forward(self, input, lengths):
        embedded = self.embedding(input)
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        outputs, (hidden, cell) = self.lstm(packed)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, tag_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(tag_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tag_size)

    def forward(self, input, hidden, cell):
        embedded = self.embedding(input)
        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        predictions = self.fc(outputs)
        return predictions, hidden, cell
    
class Encoder_Decoder_Model(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, tag_size, sos_index):
        super().__init__()
        self.encoder = Encoder(vocab_size, embed_dim, hidden_dim)
        self.decoder = Decoder(tag_size, embed_dim, hidden_dim)
        self.sos_index = sos_index
        self.device = torch.device("xpu" if torch.xpu.is_available() else "cpu")

    # def forward(self, input_seq, input_tag_real, teacher_forcing_prob=0.5):
    #     hidden, cell = self.encoder(input_seq)
    #     input_tag = torch.tensor([[self.sos_index]]*len(input_seq), device=self.device)
    #     pred_logits = torch.zeros((len(input_seq), len(input_tag_real[0]), self.decoder.fc.out_features), device=self.device)
    #     for i in range(len(input_tag_real[0])):
    #         output, hidden, cell = self.decoder(input_tag, hidden, cell)
    #         pred_logits[:, i, :] = output.squeeze(1)
    #         if random.random() < teacher_forcing_prob:
    #             input_tag = input_tag_real[:, i].unsqueeze(1)
    #         else:
    #             top1 = output.argmax(dim=-1)
    #             input_tag = top1
    #     return pred_logits, hidden, cell

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
