"""Shared utilities: tokenisation, vocabulary, data splits, CNN / BiLSTM models, training loop."""
import json, os, random, re, time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

FAMILIES = ["GPT", "Claude", "Llama", "Qwen", "Mistral", "Google"]
FAM2ID = {f: i for i, f in enumerate(FAMILIES)}
PAD, UNK, SEP = "<pad>", "<unk>", "<sep>"
DEVICE = torch.device("cpu")
torch.set_num_threads(max(1, os.cpu_count() or 1))

# ----------------------------------------------------------------------------- tokenisation
# Formatting is part of an LLM's "fingerprint", so the tokenizer keeps newlines (<nl>),
# markdown symbols and punctuation as separate tokens, and keeps letter case.
_TOK_RE = re.compile(r"<nl>|\w+|[^\w\s]")


def tokenize(text: str):
    text = text.replace("\r", "")
    text = re.sub(r"\n", " <nl> ", text)
    return _TOK_RE.findall(text)


class Vocab:
    def __init__(self, token_lists, min_freq=2, max_size=30000):
        cnt = Counter(t for toks in token_lists for t in toks)
        words = [w for w, c in cnt.most_common(max_size) if c >= min_freq]
        self.itos = [PAD, UNK, SEP] + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, toks):
        return [self.stoi.get(t, 1) for t in toks]


def make_tokens(df, mode, max_len, max_in=64):
    """mode: 'output' | 'input' | 'both'. Returns list of token lists."""
    res = []
    for inp, out in zip(df["input"], df["output"]):
        if mode == "output":
            toks = tokenize(out)[:max_len]
        elif mode == "input":
            toks = tokenize(inp)[:max_len]
        elif mode == "both":
            ti = tokenize(inp)[:max_in]
            toks = ti + [SEP] + tokenize(out)[: max_len - len(ti) - 1]
        else:
            raise ValueError(mode)
        res.append(toks if toks else [UNK])
    return res


# ----------------------------------------------------------------------------- splits
def group_split(df, seed=0, test_frac=0.2, val_frac=0.1):
    """Split by prompt so that the same prompt never appears in two splits."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trv, te = next(gss.split(df, groups=df["prompt_id"]))
    df_trv, df_te = df.iloc[trv], df.iloc[te]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_frac / (1 - test_frac), random_state=seed)
    tr, va = next(gss2.split(df_trv, groups=df_trv["prompt_id"]))
    return df_trv.iloc[tr].reset_index(drop=True), df_trv.iloc[va].reset_index(drop=True), df_te.reset_index(drop=True)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


# ----------------------------------------------------------------------------- models
class TextCNN(nn.Module):
    """Kim (2014)-style CNN: embedding -> parallel 1D convs (k=3,4,5) -> global max-pool -> FC."""

    def __init__(self, vocab_size, n_classes, emb_dim=128, n_filters=128, kernels=(3, 4, 5), dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, n_filters, k, padding=k // 2) for k in kernels])
        self.emb_drop = nn.Dropout(0.2)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(kernels), n_classes)

    def forward(self, x, lengths):
        mask = (x != 0).unsqueeze(1)                         # B x 1 x T
        e = self.emb_drop(self.emb(x)).transpose(1, 2)           # B x E x T
        pooled = []
        for conv in self.convs:
            h = F.relu(conv(e))[:, :, : x.size(1)]
            h = h.masked_fill(~mask, -1e4)                   # ignore padding in max-pool
            pooled.append(h.max(dim=2).values)
        return self.fc(self.drop(torch.cat(pooled, 1)))


class BiLSTM(nn.Module):
    """Embedding -> 1-layer bidirectional LSTM (packed) -> [mean-pool ; max-pool] -> FC."""

    def __init__(self, vocab_size, n_classes, emb_dim=128, hidden=64, dropout=0.4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.emb_drop = nn.Dropout(0.2)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(4 * hidden, n_classes)

    def forward(self, x, lengths):
        e = self.emb_drop(self.emb(x))
        packed = nn.utils.rnn.pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        mask = (x != 0).unsqueeze(2).float()
        mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        mx = out.masked_fill(mask == 0, -1e4).max(1).values
        return self.fc(self.drop(torch.cat([mean, mx], 1)))


MODELS = {"cnn": TextCNN, "lstm": BiLSTM}


# ----------------------------------------------------------------------------- training
def to_tensor(ids_list):
    lens = torch.tensor([len(x) for x in ids_list])
    X = torch.zeros(len(ids_list), int(lens.max()), dtype=torch.long)
    for i, ids in enumerate(ids_list):
        X[i, : len(ids)] = torch.tensor(ids)
    return X, lens


def batches(ids, y, bs, shuffle):
    idx = np.arange(len(ids))
    if shuffle:
        np.random.shuffle(idx)
    else:  # sort by length for faster inference
        idx = np.argsort([len(i) for i in ids])
    for s in range(0, len(idx), bs):
        b = idx[s: s + bs]
        X, L = to_tensor([ids[i] for i in b])
        yield b, X, L, torch.tensor(y[b])


@torch.no_grad()
def predict(model, ids, bs=256):
    model.eval()
    probs = np.zeros((len(ids), model.fc.out_features), dtype=np.float32)
    dummy = np.zeros(len(ids), dtype=np.int64)
    for b, X, L, _ in batches(ids, dummy, bs, shuffle=False):
        probs[b] = F.softmax(model(X, L), 1).numpy()
    return probs


def train_model(arch, train_ids, y_tr, val_ids, y_va, vocab_size, n_classes, seed=0,
                epochs=10, bs=64, lr=None, patience=3, verbose=True):
    set_seed(seed)
    lr = lr or {"cnn": 1e-3, "lstm": 2e-3}[arch]
    model = MODELS[arch](vocab_size, n_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best, best_state, bad, hist = -1, None, 0, []
    for ep in range(1, epochs + 1):
        model.train(); t0 = time.time(); tot = 0
        for _, X, L, yb in batches(train_ids, y_tr, bs, shuffle=True):
            opt.zero_grad()
            loss = F.cross_entropy(model(X, L), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += loss.item() * len(yb)
        pv = predict(model, val_ids).argmax(1)
        f1 = f1_score(y_va, pv, average="macro")
        hist.append(dict(epoch=ep, train_loss=tot / len(y_tr), val_acc=accuracy_score(y_va, pv), val_f1=f1,
                         sec=time.time() - t0))
        if verbose:
            print(f"  [{arch}] ep{ep} loss={tot/len(y_tr):.3f} val_acc={hist[-1]['val_acc']:.3f} "
                  f"val_f1={f1:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if f1 > best:
            best, bad = f1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, hist


def metrics(y, p, labels=None):
    labels = labels if labels is not None else list(range(len(FAMILIES)))
    return dict(acc=float(accuracy_score(y, p)), macro_f1=float(f1_score(y, p, average="macro", labels=labels)),
                per_class_f1=[float(v) for v in f1_score(y, p, average=None, labels=labels)],
                confusion=confusion_matrix(y, p, labels=labels).tolist())


def run_neural(arch, df_tr, df_va, df_te_dict, mode="output", max_len=256, seed=0, epochs=10, verbose=True,
               label_col="family", classes=None):
    """Train arch on df_tr (select on df_va) and evaluate on each test frame in df_te_dict."""
    classes = classes or FAMILIES
    c2i = {c: i for i, c in enumerate(classes)}
    tok_tr = make_tokens(df_tr, mode, max_len)
    vocab = Vocab(tok_tr)
    enc = lambda d: [vocab.encode(t) for t in make_tokens(d, mode, max_len)]
    y = lambda d: d[label_col].map(c2i).values
    t0 = time.time()
    model, hist = train_model(arch, [vocab.encode(t) for t in tok_tr], y(df_tr), enc(df_va), y(df_va),
                              len(vocab), len(classes), seed=seed, epochs=epochs, verbose=verbose)
    out = dict(arch=arch, mode=mode, max_len=max_len, seed=seed, vocab=len(vocab),
               n_params=sum(p.numel() for p in model.parameters()), train_time=time.time() - t0,
               history=hist, test={})
    for name, d in df_te_dict.items():
        p = predict(model, enc(d)).argmax(1)
        out["test"][name] = metrics(y(d), p, labels=list(range(len(classes))))
    return out, model, vocab


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w"), indent=1)


def load_df(path="data/llm_dataset.csv"):
    return pd.read_csv(path, keep_default_na=False)
