from transformers import AutoTokenizer
from esm_embeddings import PFGPT_HF_MODEL_PATH

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(PFGPT_HF_MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer