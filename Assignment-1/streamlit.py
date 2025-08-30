import streamlit as st
from HMM.HMM import get_POS_tags
from Enc_Dec.enc_dec_lstm import Encoder_Decoder_Model
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from nltk.tokenize import word_tokenize

### ------------------------------- HMM Model Setup ------------------------------- ###

# Load the transition probabilities
with open("HMM/data/tag_given_tag.json") as f:
    tag_given_tag = json.load(f)

# Load the emission probabilities
with open("HMM/data/word_given_tag.json") as f:
    word_given_tag = json.load(f)

# Load the total count of each tag
with open("HMM/data/total_given_tag.json") as f:
    total_given_tag = json.load(f)

# Load the tag to index mapping
with open("HMM/data/tag_to_index.json") as f:
    tag_to_index = json.load(f)

### ------------------------------- Encoder-Decoder Model Setup ------------------------------- ###
epochs = 20
embed_dim = 300
hidden_dim = 256
batch_size = 128
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")

with open("Enc_Dec/tokenizer/tag2idx.json") as f:
    tag2idx = json.load(f)
    idx2tag = {v: k for k, v in tag2idx.items()}

with open("Enc_Dec/tokenizer/word2idx.json") as f:
    word2idx = json.load(f)

with open("Enc_Dec/tokenizer/word_embeddings.pt", "rb") as f:
    embedding_matrix = torch.load(f)

tag_size = len(tag2idx)
vocab_size = len(word2idx)

if st.session_state.get("model_encdec") is None:
    model_encdec = Encoder_Decoder_Model(
        vocab_size, embed_dim, hidden_dim, tag_size, tag2idx["<SOS>"], embedding_matrix
    ).to(device)
    lr = 0.001
    model_encdec.load_state_dict(
        torch.load(
            f"Enc_Dec/models/encoder_decoder_model_{lr}lr_{batch_size}bs_{epochs}epochs.pth",
            map_location=device,
        )
    )
    st.session_state.model_encdec = model_encdec
else:
    model_encdec = st.session_state.model_encdec

### ------------------------------- LLM Setup ------------------------------- ###
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
if st.session_state.get("model_llm") is None:
    name = "TweebankNLP/bertweet-tb2_ewt-pos-tagging"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model_llm = AutoModelForTokenClassification.from_pretrained(name)
    generator = pipeline(
        task="token-classification", model=name, tokenizer=name, device=device
    )
    st.session_state.model_llm = generator
else:
    generator = st.session_state.model_llm

with open("LLM/upos_to_utag.json") as f:
    upos_to_utag = json.load(f)

# Title for the app
st.title("Part-of-Speech Tagging")

model_type = st.selectbox(
    "Choose your model for POS tagging",
    ("HMM", "Encoder-Decoder (Greedy)", "Encoder-Decoder (Beam Search)", "LLM"),
)

# Input text box
input_text = st.text_input(
    placeholder="Type here...", label="Enter a sentence to get its POS tags:"
)

# Button to trigger POS tagging
if input_text:
    words = [w.lower() for w in word_tokenize(input_text)]
    if model_type == "HMM":
        print("Using HMM model for POS tagging")
        pos_tags = get_POS_tags(
            words, tag_given_tag, word_given_tag, total_given_tag, tag_to_index
        )

    elif model_type == "Encoder-Decoder (Greedy)":
        print("Using Encoder-Decoder model for POS tagging")
        word_ids = torch.tensor(
            [word2idx.get(w, 1) for w in words], device=device
        ).unsqueeze(0)
        length = [word_ids.shape[1]]
        with torch.no_grad():
            output_tags = model_encdec.generate(word_ids, word_ids.shape[1], length)
            pos_tags = [
                [idx2tag[idx.item()] for idx in output_tags[i]]
                for i in range(len(output_tags))
            ][0][1:]

    elif model_type == "Encoder-Decoder (Beam Search)":
        print("Using Encoder-Decoder model with Beam Search for POS tagging")
        word_ids = torch.tensor(
            [word2idx.get(w, 1) for w in words], device=device
        ).unsqueeze(0)
        length = [word_ids.shape[1]]
        with torch.no_grad():
            output_tags = model_encdec.generate_beam_search(
                word_ids, word_ids.shape[1], length
            )
            pos_tags = [
                [idx2tag[idx.item()] for idx in output_tags[i]]
                for i in range(len(output_tags))
            ][0][1:]

    elif model_type == "LLM":
        print("Using LLM model for POS tagging")
        outputs = generator(" ".join(words))
        pos_tags = [
            output["entity"] for output in outputs if not output["word"].endswith("@@")
        ]
        pos_tags = [upos_to_utag[tag] for tag in pos_tags]

    st.divider()

    # Display the result as a dictionary
    st.write("POS Tags for the input sentence with model:", model_type)

    df = pd.DataFrame({"Word": word_tokenize(input_text), "POS Tag": pos_tags})
    df.reset_index(drop=True, inplace=True)
    st.table(df)
