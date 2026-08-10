from sentence_transformers import SentenceTransformer
import streamlit as st
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, AutoModel
import os
import pickle
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig
from torch.utils.data import DataLoader
from prompt import PROMPT_INSTRUCT
from test_instruct import generate as generate_instruct
from test_instruct_svdd import generate as generate_instruct_svdd
import numpy as np
from box import Box

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.title("Hate Speech to Neutral Text Conversion")

col1, col2, col3 = st.columns([2, 2, 1], gap="medium")
epochs = 3
batch_size = 8
lr = 2e-4
num_unmask_steps = 128
gen_length = 128
with col1:
    cfg_scale = st.slider("CFG Scale", 1.0, 1.8, 1.0, 0.4)
p_uncond = 0.0
with col2:
    checkpoint_num = st.slider("Checkpoint Number", 2321, 6963, 2321, 2321)
with col3:
    do_svdd = st.toggle("SVDD", value=True)
svdd_numcands = 4

model_name = os.path.join(
                "models_instruct copy",
                f"llada_{epochs}ep-{batch_size}bs-{lr}lr-{p_uncond}puncond",
                f"checkpoint-{checkpoint_num}",
            )

if st.session_state.get("tokenizer") is None:
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    
    # The LLaDA architecture theoretically supports both left-padding and right-padding. 
    # However, the sampling code implementation is simpler with left-padding.
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'

    # If the padding ID equals the mask ID, you need to modify our generate function to achieve correct inference.
    assert tokenizer.pad_token_id != 126336
    
    st.session_state["tokenizer"] = tokenizer
else:
    tokenizer = st.session_state["tokenizer"]

if st.session_state.get("model") is None or st.session_state.get("model_name") != model_name:
    if st.session_state.get("model") is not None:
        del st.session_state["model"]
    torch.cuda.empty_cache()
    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    print("Loading PEFT model from ", model_name)
    model = PeftModel.from_pretrained(
        model,
        model_name,
    )
    model.eval()
    st.session_state["model"] = model
    st.session_state["model_name"] = model_name
else:
    model = st.session_state["model"]

if st.session_state.get("sentence_model") is None:
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    st.session_state["sentence_model"] = sentence_model
else:
    sentence_model = st.session_state["sentence_model"]

# model_type = st.selectbox(
#     "Choose your model for POS tagging",
#     # ("HMM", "Encoder-Decoder (Greedy)", "Encoder-Decoder (Beam Search)", "LLM"),
#     # ("HMM", "Encoder-Decoder (Greedy)", "LLM"),
#     ("Rule Based", "LSTM (Greedy)", "LSTM (Beam Search)", "Transformer (Greedy)", "Transformer (Beam Search)", "LLM"),
# )

# Input text box
input_text = st.text_input(
    placeholder="Type Hate Speech here...", label="Enter a hate speech text to convert to neutral text"
)

# Button to trigger POS tagging
if input_text:
    # words = [w.lower() for w in input_text.split()]
    prompt = PROMPT_INSTRUCT.format(hate_text=input_text, neutral_text="")[:-1]

    encoded_outputs = tokenizer(
        [prompt],
        add_special_tokens=False,
        padding=True,
        return_tensors="pt"
    )
    input_ids = encoded_outputs['input_ids'].to(device)
    attention_mask = encoded_outputs['attention_mask'].to(device)

    with st.spinner("Generating neutral text..."):
        if do_svdd:
            output = generate_instruct_svdd(
                model,
                input_ids,
                attention_mask,
                steps=num_unmask_steps,
                gen_length=gen_length,
                block_length=gen_length,
                temperature=0.0,
                cfg_scale=cfg_scale-1,
                remasking="random",
                sentence_model=sentence_model,
                tokenizer=tokenizer,
                num_candidates=svdd_numcands,
                hates=[input_text],
            )
        else:
            output = generate_instruct(
                model,
                input_ids,
                attention_mask,
                steps=num_unmask_steps,
                gen_length=gen_length,
                block_length=gen_length,
                temperature=0.0,
                cfg_scale=cfg_scale,
                remasking="low_confidence",
            )

    output_text = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    print("Raw output text:", output_text)
    output_text = output_text.split('\n')[0]
    print("Hate speech:", input_text)
    
    st.text_input("Neutral Text", value=output_text)

    # --- compute metrics for this single input/output pair similar to eval.py ---
    try:
        from paradetox.evaluation_detox.metric_tools.style_transfer_accuracy import classify_preds
        from paradetox.evaluation_detox.metric_tools.content_similarity import flair_sim
        from paradetox.evaluation_detox.metric_tools.fluency import cola_fluency
        from paradetox.evaluation_detox.metric_tools.joint_metrics import get_j
    except Exception as e:
        st.warning("paradetox metric tools not available: " + str(e))
    else:
        metric_args = Box({
            'batch_size': 32,
            'cola_classifier_path': '/content/drive/MyDrive/style_transfer/cola_classifier',
            'wieting_tokenizer_path': 'sim.sp.30k.model',
            'wieting_model_path': 'sim.pt',
            't1': 75., 't2': 70., 't3': 12.
        })

        # helper to convert torch tensors / scalars to numpy
        def to_numpy(x):
            if isinstance(x, torch.Tensor):
                try:
                    return x.detach().cpu().numpy()
                except Exception:
                    return np.array(x.item())
            return np.array(x)

        preds = [output_text.lower()]
        hates = [input_text.lower()]

        with st.spinner("Computing evaluation metrics..."):
            # Style Transfer Accuracy of Inputs
            accuracy_by_sent = classify_preds(metric_args, hates)
            STA_IN = float(np.mean(to_numpy(accuracy_by_sent)))

            # Style Transfer Accuracy of Outputs
            accuracy_by_sent = classify_preds(metric_args, preds)
            STA_OUT = float(np.mean(to_numpy(accuracy_by_sent)))

            # Content similarity
            emb_sim_stats = flair_sim(metric_args, hates, preds)
            SIM = float(np.mean(to_numpy(emb_sim_stats)))

            # Fluency of Inputs
            cola_stats = cola_fluency(hates)
            FL_IN = float(sum(cola_stats) / len(hates))

            # Fluency of Outputs
            cola_stats = cola_fluency(preds)
            FL_OUT = float(sum(cola_stats) / len(preds))

            # Fluency similarity
            FL_acc = sum(np.array(cola_fluency(preds)) == np.array(cola_fluency(hates)))/len(preds)

        st.subheader("Evaluation metrics (input -> output)")
        # build a small table and display
        metrics = {
            "STA_IN": STA_IN,
            "STA_OUT": STA_OUT,
            "SIM": SIM,
            "FL_IN": FL_IN,
            "FL_OUT": FL_OUT,
            "FL_acc": FL_acc,
            # "J": J,
            # "J_with_FL_acc": J_with_FL_acc
        }
        def fmt_val(v):
            return round(float(v), 4)

        df = pd.DataFrame(
            [(k, fmt_val(v)) for k, v in metrics.items()],
            columns=["Metric", "Value"]
        )
        st.table(df)


