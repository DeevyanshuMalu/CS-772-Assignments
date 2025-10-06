import streamlit as st
from Rule_Based.utils import transliterate_en_to_hi
from LSTM import models as lstm_models
from LSTM import dataset as lstm_dataset
from Transformers import models as transformer_models
from Transformers import dataset as transformer_dataset
from LLM.prompts import SYSTEM_PROMPT, USER_PROMPT
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from nltk.tokenize import word_tokenize
import os
import pickle
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### ------------------------------- LSTM Model Setup ------------------------------- ###
lr = 1e-4
num_epochs = 100
teacher_forcing_ratio = 0.5
epoch_to_test = 30
embed_size = 128
hidden_size = 256
num_layers = 2
dropout = 0.2
model_dir = os.path.join(
    "LSTM",
    "models",
    f"{lr}lr_{num_epochs}epochs_{teacher_forcing_ratio}tf",
)
vocab_path = os.path.join("LSTM", "models", "vocab.pkl")
with open(vocab_path, "rb") as f:
    vocab = pickle.load(f)
src_vocab_LSTM = vocab["src_vocab"]
tgt_vocab_LSTM = vocab["tgt_vocab"]
if st.session_state.get("model_lstm") is None:
    lstm_model = lstm_models.Seq2SeqModel(
        src_vocab_size=len(src_vocab_LSTM),
        tgt_vocab_size=len(tgt_vocab_LSTM),
        embed_size=embed_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    model_path = os.path.join(model_dir, f"checkpoint_epoch_{epoch_to_test}.pt")
    state_dict = torch.load(model_path, map_location=device)
    lstm_model.load_state_dict(state_dict["model_state_dict"])
    lstm_model.eval()
    st.session_state.model_lstm = lstm_model
else:
    lstm_model = st.session_state.model_lstm

### ------------------------------- Transformer Model Setup ------------------------------- ###
lr = 1e-4
num_epochs = 50
epoch_to_test = 30
d_model = 256
nhead = 8
num_encoder_layers = 2
num_decoder_layers = 2
dim_feedforward = 1024
dropout = 0.1
max_len = 50
dropout = 0.2
model_dir = os.path.join(
    "Transformers",
    "models",
    f"{lr}lr_{num_epochs}epochs",
)
vocab_path = os.path.join("Transformers", "models", "vocab.pkl")
with open(vocab_path, "rb") as f:
    vocab = pickle.load(f)
src_vocab_transformers = vocab["src_vocab"]
tgt_vocab_transformers = vocab["tgt_vocab"]
if st.session_state.get("model_transformers") is None:
    transformer_model = transformer_models.TransformerModel(
        input_vocab_size=len(src_vocab_transformers),  # Source vocab size
        output_vocab_size=len(tgt_vocab_transformers),  # Target vocab size
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_len=max_len,
        pad_token_id=0,  # <PAD>
        sos_token_id=1,  # <SOS>
        eos_token_id=2,  # <EOS>
        unk_token_id=3,  # <UNK>
    ).to(device)
    model_path = os.path.join(model_dir, f"checkpoint_epoch_{epoch_to_test}.pt")
    state_dict = torch.load(model_path, map_location=device)
    transformer_model.load_state_dict(state_dict["model_state_dict"])
    transformer_model.eval()
    st.session_state.model_transformers = transformer_model
else:
    transformer_model = st.session_state.model_transformers

### ------------------------------- LLM Setup ------------------------------- ###
if st.session_state.get("model_llm") is None:
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    st.session_state.model_llm = pipe
else:
    pipe = st.session_state.model_llm

# Title for the app
st.title("English to Hindi Transliteration")

model_type = st.selectbox(
    "Choose your model for POS tagging",
    # ("HMM", "Encoder-Decoder (Greedy)", "Encoder-Decoder (Beam Search)", "LLM"),
    # ("HMM", "Encoder-Decoder (Greedy)", "LLM"),
    ("Rule Based", "LSTM (Greedy)", "LSTM (Beam Search)", "Transformer (Greedy)", "Transformer (Beam Search)", "LLM"),
)

if model_type == "LLM":
    temperature = st.slider("Temperature", 0.0, 2.0, 1.0, 0.1)
    top_p = st.slider("Top-p (nucleus sampling)", 0.0, 1.0, 1.0, 0.05)

# Input text box
input_text = st.text_input(
    placeholder="Type here...", label="Enter a sentence to get its transliteration:"
)

# Button to trigger POS tagging
if input_text:
    words = [w.lower() for w in input_text.split()]
    if model_type == "Rule Based":
        print("Using Rule Based model for POS tagging")
        transliterated_words = [transliterate_en_to_hi(w) for w in words]

    elif "LSTM" in model_type or "Transformer" in model_type:
        if "LSTM" in model_type:
            print("Using LSTM model for POS tagging")
        elif "Transformer" in model_type:
            print("Using Transformer model for POS tagging")
        if "LSTM" in model_type:
            src_vocab = src_vocab_LSTM
            tgt_vocab = tgt_vocab_LSTM
            test_dataset = lstm_dataset.TransliterationDataset(
                [(w, "") for w in words], src_vocab, tgt_vocab, max_len=50
            )
        elif "Transformer" in model_type:
            src_vocab = src_vocab_transformers
            tgt_vocab = tgt_vocab_transformers
            test_dataset = transformer_dataset.TransliterationDataset(
                [(w, "") for w in words], src_vocab, tgt_vocab, max_len=50
            )
        test_loader = DataLoader(test_dataset, batch_size=len(words), shuffle=False)
        transliterated_words = []
        for batch in test_loader:
            src = batch["src"].to(device)
            src_lens = batch["src_len"]
            with torch.no_grad():
                if model_type == "LSTM (Greedy)":
                    output_tokens = lstm_model.generate_greedy(src, src_lens)
                elif model_type == "LSTM (Beam Search)":
                    output_tokens = lstm_model.generate_batch_beam_search(
                        src, src_lens, beam_size=5
                    )
                elif model_type == "Transformer (Greedy)":
                    output_tokens = transformer_model.generate_greedy(src)
                elif model_type == "Transformer (Beam Search)":
                    output_tokens = transformer_model.generate_batch_beam_search(
                        src, beam_size=5
                    )
            src_idx2char = test_dataset.src_idx2char
            tgt_idx2char = test_dataset.tgt_idx2char
            for i in range(src.size(0)):
                transliterated_words.append(
                    "".join(
                        [
                            tgt_idx2char[idx.item()]
                            for idx in output_tokens[i]
                            if tgt_idx2char[idx.item()] not in ["<PAD>", "<SOS>", "<EOS>"]
                        ]
                    )
                )
    elif model_type == "LLM":
        print("Using LLM model for POS tagging")
        transliterated_words = []
        for w in words:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(input=w)},
            ]
            outputs = pipe(
                messages,
                max_new_tokens=256,
                temperature=temperature,
                top_p=top_p,
                return_full_text=False,
            )
            pred = outputs[0]["generated_text"].strip()
            transliterated_words.append(pred)

    st.divider()

    # Display the result as a dictionary
    st.write("Transliteration for the input sentence with model:", model_type)

    df = pd.DataFrame({"Word": words, "Transliteration": transliterated_words})
    df.reset_index(drop=True, inplace=True)
    st.table(df)
