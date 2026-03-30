import streamlit as st
import torch
from pathlib import Path

from config import get_config, latest_weights_file_path
from model import build_transformer
from tokenizers import Tokenizer

# --- Setup ---
st.set_page_config(page_title="Transformer Translator", layout="centered")

st.title("EN → IT Translator (Transformer from Scratch)")
st.markdown("Real-time translation using a GPT-style transformer trained on OPUS Books dataset.")

# Load config
config = get_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    tokenizer_src = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_src']))))
    tokenizer_tgt = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_tgt']))))

    model = build_transformer(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        config["seq_len"],
        config["seq_len"],
        d_model=config["d_model"]
    ).to(device)

    model_path = latest_weights_file_path(config)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    return model, tokenizer_src, tokenizer_tgt

model, tokenizer_src, tokenizer_tgt = load_model()

# --- Inference ---
def translate(sentence):
    seq_len = config['seq_len']

    source = tokenizer_src.encode(sentence)
    source = torch.cat([
        torch.tensor([tokenizer_src.token_to_id('[SOS]')]),
        torch.tensor(source.ids),
        torch.tensor([tokenizer_src.token_to_id('[EOS]')]),
        torch.tensor([tokenizer_src.token_to_id('[PAD]')] * (seq_len - len(source.ids) - 2))
    ]).unsqueeze(0).to(device)

    source_mask = (source != tokenizer_src.token_to_id('[PAD]')).unsqueeze(1).unsqueeze(1)

    encoder_output = model.encode(source, source_mask)

    decoder_input = torch.tensor([[tokenizer_tgt.token_to_id('[SOS]')]]).to(device)

    for _ in range(seq_len):
        decoder_mask = torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(1))), diagonal=1).type(torch.bool).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)

        decoder_input = torch.cat([decoder_input, next_word.unsqueeze(0)], dim=1)

        if next_word.item() == tokenizer_tgt.token_to_id('[EOS]'):
            break

    return tokenizer_tgt.decode(decoder_input.squeeze().tolist())


# --- UI ---
input_text = st.text_area("Enter English text:", "Hello, how are you?")

if st.button("Translate"):
    with st.spinner("Translating..."):
        output = translate(input_text)
    st.success("Translation:")
    st.write(output)

st.markdown("---")
st.caption(f"Running on: {device}")