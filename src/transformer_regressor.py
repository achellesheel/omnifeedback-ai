"""V3: transfer-learned urgency regressor.

V2's BiLSTM learns its own embedding table from scratch on ~4,000 rows, which
caps its working vocabulary at ~196 words seen in training — any real-world
phrasing outside that narrow set collapses to near-meaningless <UNK> tokens
(see PROGRESS.md Milestone 8 for the diagnosis: 40-60% OOV rate on natural
customer text, critical and positive examples scoring nearly identically).

V3 instead fine-tunes a pretrained transformer encoder (DistilBERT) that
already has general-purpose English language understanding from pretraining
on a massive corpus. Subword tokenization means there is no closed-vocabulary
OOV problem at all — any English word decomposes into known subword pieces.
We freeze the embeddings and the first N transformer layers (analogous to
freezing an ImageNet-pretrained CNN backbone and fine-tuning only the last
few layers + head) and only train the last few layers plus a small regression
head, so most of the ~66M parameters carry over pretrained knowledge rather
than being relearned from a small dataset.
"""
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
BASE_MODEL_NAME = "distilbert-base-uncased"
N_FROZEN_LAYERS = 4  # DistilBERT has 6 transformer layers; freeze the first 4, fine-tune the last 2 + head


class UrgencyTextDataset(Dataset):
    def __init__(self, texts, scores, tokenizer, max_len=64):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding="max_length", max_length=max_len, return_tensors="pt"
        )
        self.scores = torch.tensor(list(scores), dtype=torch.float32)

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["score"] = self.scores[idx]
        return item


class TransformerUrgencyRegressor(nn.Module):
    def __init__(self, base_model_name=BASE_MODEL_NAME, n_frozen_layers=N_FROZEN_LAYERS, dropout=0.2):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(base_model_name)

        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
        for layer in self.encoder.transformer.layer[:n_frozen_layers]:
            for param in layer.parameters():
                param.requires_grad = False

        hidden_size = self.encoder.config.dim
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # mean-pool over non-padding tokens (more robust than [CLS]-only for short, varied text)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return torch.sigmoid(self.head(self.dropout(pooled))).squeeze(-1)

    def trainable_param_fraction(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable / total, trainable, total

    def trainable_state_dict(self):
        """Only the fine-tuned subset (last N layers + head) — the frozen pretrained backbone
        is re-downloaded from HuggingFace at load time instead of being persisted, since it's
        never modified. Keeps the saved checkpoint ~55MB instead of ~265MB (the full model would
        exceed GitHub's 100MB per-file push limit and require Git LFS for no benefit)."""
        trainable_names = {name for name, p in self.named_parameters() if p.requires_grad}
        return {k: v for k, v in self.state_dict().items() if k in trainable_names}


def train_transformer_regressor(
    df, text_col="raw_text", target_col="urgency_score",
    epochs=8, batch_size=16, lr=2e-5, weight_decay=1e-5, patience=3,
    max_len=64, val_frac=0.15, test_frac=0.15, save=True, device=None, model_dir=None,
):
    """Same train/val/test split methodology (seed, fractions) as dl_lstm.train_bilstm_model,
    so the held-out test rows are identical and V2 vs V3 metrics are directly comparable.

    Trains on `raw_text` rather than V2's heavily regex-cleaned `cleaned_text` by design:
    DistilBERT's subword tokenizer and pretrained knowledge benefit from real punctuation and
    casing (e.g. "!!!", capitalization) as signal, whereas V2's from-scratch small-vocab BiLSTM
    needed that noise stripped out. Different architectures, different right preprocessing.
    """
    import numpy as np
    from transformers import AutoTokenizer

    model_dir = model_dir or MODELS_DIR
    device = device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    texts = df[text_col].tolist()
    scores = df[target_col].tolist()
    n = len(texts)
    idx = np.random.RandomState(42).permutation(n)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test_idx, val_idx, train_idx = idx[:n_test], idx[n_test:n_test + n_val], idx[n_test + n_val:]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    def make_loader(indices, shuffle):
        t = [texts[i] for i in indices]
        s = [scores[i] for i in indices]
        ds = UrgencyTextDataset(t, s, tokenizer, max_len=max_len)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(train_idx, True)
    val_loader = make_loader(val_idx, False)
    test_loader = make_loader(test_idx, False)

    model = TransformerUrgencyRegressor().to(device)
    frac, trainable, total = model.trainable_param_fraction()
    print(f"Trainable params: {trainable:,} / {total:,} ({frac*100:.1f}%)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["score"].to(device)
            optimizer.zero_grad()
            preds = model(input_ids, attention_mask)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                y = batch["score"].to(device)
                val_losses.append(criterion(model(input_ids, attention_mask), y).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"epoch {epoch}: train={train_loss:.5f} val={val_loss:.5f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.load_state_dict(best_state)

    model.eval()
    test_losses, test_preds, test_targets = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["score"].to(device)
            preds = model(input_ids, attention_mask)
            test_losses.append(criterion(preds, y).item())
            test_preds.extend(preds.cpu().numpy().tolist())
            test_targets.extend(y.cpu().numpy().tolist())
    test_mse = float(np.mean(test_losses)) if test_losses else None

    binary_preds = [0 if p >= 0.5 else 1 for p in test_preds]
    binary_targets = [0 if t >= 0.5 else 1 for t in test_targets]
    from sklearn.metrics import f1_score
    test_f1 = float(f1_score(binary_targets, binary_preds, average="macro")) if test_targets else None

    if save:
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.trainable_state_dict(), model_dir / "transformer_urgency.pth")
        tokenizer.save_pretrained(model_dir / "tokenizer")
        with open(model_dir / "model_meta.json", "w") as f:
            json.dump({"base_model_name": BASE_MODEL_NAME, "n_frozen_layers": N_FROZEN_LAYERS, "max_len": max_len}, f)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "history": history,
        "best_val_loss": best_val_loss,
        "test_mse": test_mse,
        "test_f1": test_f1,
        "stopped_epoch": len(history["train_loss"]),
        "trainable_fraction": frac,
    }


def load_transformer_regressor(device=None, model_dir=None):
    from transformers import AutoTokenizer

    model_dir = model_dir or MODELS_DIR
    device = device or ("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    with open(model_dir / "model_meta.json") as f:
        meta = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    model = TransformerUrgencyRegressor(base_model_name=meta["base_model_name"], n_frozen_layers=meta["n_frozen_layers"]).to(device)
    trainable_state = torch.load(model_dir / "transformer_urgency.pth", map_location=device, weights_only=True)
    model.load_state_dict(trainable_state, strict=False)  # frozen backbone comes from from_pretrained() above
    model.eval()
    return model, tokenizer, device, meta.get("max_len", 64)


def predict_urgency_transformer(model, tokenizer, text, device, max_len=64):
    enc = tokenizer([text], truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        score = model(input_ids, attention_mask).item()
    return float(score)
