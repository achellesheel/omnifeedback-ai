"""PyTorch BiLSTM continuous urgency regressor: vocab, Dataset/DataLoader, model, training."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PAD_TOKEN, UNK_TOKEN = "<PAD>", "<UNK>"


class TextVocabulary:
    def __init__(self, texts: list, min_freq: int = 2, max_vocab: int = 10000):
        counter = Counter()
        for t in texts:
            counter.update(t.split())
        self.itos = [PAD_TOKEN, UNK_TOKEN] + [
            w for w, c in counter.most_common(max_vocab) if c >= min_freq
        ]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, text: str, max_len: int = 128) -> list:
        tokens = text.split()[:max_len]
        ids = [self.stoi.get(tok, self.stoi[UNK_TOKEN]) for tok in tokens]
        ids += [self.stoi[PAD_TOKEN]] * (max_len - len(ids))
        return ids

    def to_dict(self) -> dict:
        return {"itos": self.itos}

    @classmethod
    def from_dict(cls, d: dict):
        vocab = cls.__new__(cls)
        vocab.itos = d["itos"]
        vocab.stoi = {w: i for i, w in enumerate(vocab.itos)}
        return vocab


class FeedbackDataset(Dataset):
    def __init__(self, texts: list, scores: list, vocab: TextVocabulary, max_len: int = 128):
        self.encoded = [vocab.encode(t, max_len) for t in texts]
        self.scores = scores

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        x = torch.tensor(self.encoded[idx], dtype=torch.long)
        y = torch.tensor(self.scores[idx], dtype=torch.float32)
        return x, y


class BiLSTMUrgencyRegressor(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        _, (h_n, _) = self.lstm(embedded)
        h_cat = torch.cat((h_n[-2], h_n[-1]), dim=1)  # last fwd + bwd hidden states
        out = self.dropout(h_cat)
        return self.sigmoid(self.fc(out)).squeeze(-1)


def train_bilstm_model(
    df, text_col: str = "cleaned_text", target_col: str = "urgency_score",
    epochs: int = 30, batch_size: int = 32, lr: float = 1e-3, weight_decay: float = 1e-5,
    patience: int = 4, max_len: int = 128, val_frac: float = 0.15, test_frac: float = 0.15,
    save: bool = True, device: str = None, model_dir: Path = None,
):
    model_dir = model_dir or MODELS_DIR
    """Trains with an explicit train/val/test split and early stopping on val MSE
    to avoid overfitting — the val loss curve + stopping epoch are logged and returned
    so the run can be audited instead of trusting a single final-epoch number."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    texts = df[text_col].tolist()
    scores = df[target_col].tolist()

    n = len(texts)
    idx = np.random.RandomState(42).permutation(n)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test_idx, val_idx, train_idx = idx[:n_test], idx[n_test:n_test + n_val], idx[n_test + n_val:]

    train_texts = [texts[i] for i in train_idx]
    vocab = TextVocabulary(train_texts)

    def make_loader(indices, shuffle):
        t = [texts[i] for i in indices]
        s = [scores[i] for i in indices]
        ds = FeedbackDataset(t, s, vocab, max_len=max_len)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(train_idx, True)
    val_loader = make_loader(val_idx, False)
    test_loader = make_loader(test_idx, False)

    model = BiLSTMUrgencyRegressor(vocab_size=len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_losses.append(criterion(model(x), y).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

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
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            test_losses.append(criterion(preds, y).item())
            test_preds.extend(preds.cpu().numpy().tolist())
            test_targets.extend(y.cpu().numpy().tolist())
    test_mse = float(np.mean(test_losses)) if test_losses else None

    # Dataset convention: urgency_score >= 0.5 => critical (label 0), else non-critical (label 1)
    binary_preds = [0 if p >= 0.5 else 1 for p in test_preds]
    binary_targets = [0 if t >= 0.5 else 1 for t in test_targets]
    from sklearn.metrics import f1_score
    test_f1 = float(f1_score(binary_targets, binary_preds, average="macro")) if test_targets else None

    if save:
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_dir / "bilstm_urgency.pth")
        with open(model_dir / "vocab.json", "w") as f:
            json.dump(vocab.to_dict(), f)

    return {
        "model": model,
        "vocab": vocab,
        "history": history,
        "best_val_loss": best_val_loss,
        "test_mse": test_mse,
        "test_f1": test_f1,
        "stopped_epoch": len(history["train_loss"]),
    }


def load_bilstm_model(device: str = None, model_dir: Path = None):
    model_dir = model_dir or MODELS_DIR
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    with open(model_dir / "vocab.json") as f:
        vocab = TextVocabulary.from_dict(json.load(f))
    model = BiLSTMUrgencyRegressor(vocab_size=len(vocab)).to(device)
    model.load_state_dict(torch.load(model_dir / "bilstm_urgency.pth", map_location=device, weights_only=True))
    model.eval()
    return model, vocab, device


def predict_urgency(model, vocab, text: str, device: str, max_len: int = 128) -> float:
    ids = torch.tensor([vocab.encode(text, max_len)], dtype=torch.long).to(device)
    with torch.no_grad():
        score = model(ids).item()
    return float(score)
