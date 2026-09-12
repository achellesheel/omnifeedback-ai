"""Unified urgency-scoring interface across model backends (BiLSTM for V1/V2,
transfer-learned DistilBERT for V3), so pages don't need to know which one a
variant uses — they just call load_urgency_model(variant) / score(bundle, text).
"""
from src.dl_lstm import load_bilstm_model, predict_urgency
from src.transformer_regressor import load_transformer_regressor, predict_urgency_transformer


def load_urgency_model(variant: dict):
    if variant.get("model_type") == "transformer":
        model, tokenizer, device, max_len = load_transformer_regressor(model_dir=variant["model_dir"])
        return {"type": "transformer", "model": model, "tokenizer": tokenizer, "device": device, "max_len": max_len}
    model, vocab, device = load_bilstm_model(model_dir=variant["model_dir"])
    return {"type": "bilstm", "model": model, "vocab": vocab, "device": device}


def score_urgency(bundle: dict, text: str, cleaned_text: str = None) -> float:
    """`cleaned_text` (regex-cleaned) is used for BiLSTM; transformer models score
    the raw text directly since their pretrained tokenizer benefits from real
    punctuation/casing that the BiLSTM's cleaning pipeline strips out."""
    if bundle["type"] == "transformer":
        return predict_urgency_transformer(bundle["model"], bundle["tokenizer"], text, bundle["device"], bundle["max_len"])
    return predict_urgency(bundle["model"], bundle["vocab"], cleaned_text or text, bundle["device"])
