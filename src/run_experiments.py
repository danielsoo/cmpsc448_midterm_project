"""Run the neural (CNN / BiLSTM) experiments for RQ1-RQ4.

Usage:  python src/run_experiments.py EXP_NAME [EXP_NAME ...]
        python src/run_experiments.py --list
Each experiment writes results/<EXP_NAME>.json. Existing results are skipped.
"""
import argparse, os, random, re, sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.dirname(__file__))
from common import *  # noqa

RES = "results"
SOURCES = ["helpful_base", "koala", "oasst", "selfinstruct", "vicuna"]
CODE_RE = re.compile(r"```|\b(?:code|coding|python|javascript|java|c\+\+|c#|function|sql|html|css|regex|"
                     r"program|programming|script|algorithm|bash|latex|json|api|compile|debug)\b", re.I)


# ----------------------------------------------------------------------------- text ablations (RQ4)
_STOP = set(ENGLISH_STOP_WORDS)


def strip_format(t):
    t = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", t)               # headers
    t = re.sub(r"(?m)^\s*([-*+•]|\d+[.)])\s+", "", t)     # bullets / numbered lists
    t = re.sub(r"\*\*|__|`+", "", t)                            # bold / code ticks
    t = re.sub(r"\s+", " ", t)                                   # newlines -> spaces
    return t.strip()


def drop_opener(t):
    t = t.strip()
    parts = t.split("\n", 1)
    if len(parts) == 2 and len(parts[0]) < 200:
        return parts[1].strip()
    s = re.split(r"(?<=[.!?:])\s+", t, maxsplit=1)
    return s[1] if len(s) == 2 else t


def function_words_only(t):
    return re.sub(r"\w+", lambda m: m.group(0) if m.group(0).lower() in _STOP else "W", t)


def content_words_only(t):
    words = re.findall(r"[A-Za-z]+", t.lower())
    return " ".join(w for w in words if w not in _STOP)


def shuffle_words(t, rng=random.Random(0)):
    toks = tokenize(t)
    rng.shuffle(toks)
    return " ".join(toks).replace("<nl>", "\n")


ABLATIONS = {"noformat": strip_format, "noopener": drop_opener, "funcwords": function_words_only,
             "contentonly": content_words_only, "shuffle": shuffle_words}


# ----------------------------------------------------------------------------- confounded data (RQ2)
def confounded_dataset(df, seed=0, p_match=0.9, p_other=0.15):
    """Simulate a *biased collection*: each family is mostly queried with prompts of one topic cluster."""
    prompts = df.drop_duplicates("prompt_id")[["prompt_id", "input"]].reset_index(drop=True)
    X = TfidfVectorizer(stop_words="english", min_df=2).fit_transform(prompts["input"])
    km = KMeans(6, n_init=10, random_state=seed).fit(X)
    D = km.transform(X)                              # distance of each prompt to each topic centroid
    cap = int(np.ceil(len(prompts) / 6))             # balanced topic clusters (greedy capacity assignment)
    assign, used = -np.ones(len(prompts), int), np.zeros(6, int)
    for flat in np.argsort(D, axis=None):
        i, c = divmod(int(flat), 6)
        if assign[i] < 0 and used[c] < cap:
            assign[i] = c; used[c] += 1
    prompts["cluster"] = assign
    cl = dict(zip(prompts.prompt_id, prompts.cluster))
    rng = np.random.RandomState(seed)
    keep = []
    for i, r in df.iterrows():
        match = cl[r.prompt_id] == FAM2ID[r.family]
        keep.append(rng.rand() < (p_match if match else p_other))
    d = df[np.array(keep)].copy()
    d["cluster"] = d.prompt_id.map(cl)
    return d.reset_index(drop=True)


# ----------------------------------------------------------------------------- experiment registry
def split_with_val(df_train_pool, seed=0, val_frac=0.12):
    g = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    a, b = next(g.split(df_train_pool, groups=df_train_pool.prompt_id))
    return df_train_pool.iloc[a].reset_index(drop=True), df_train_pool.iloc[b].reset_index(drop=True)


def experiments():
    E = {}
    for arch in ["cnn", "lstm"]:
        for s in [0, 1, 2]:
            E[f"rq1_{arch}_s{s}"] = dict(kind="main", arch=arch, mode="output", seed=s)
        for m in ["input", "both"]:
            E[f"rq2_{arch}_{m}"] = dict(kind="main", arch=arch, mode=m, seed=0)
        for m in ["input", "output", "both"]:
            E[f"rq2conf_{arch}_{m}"] = dict(kind="conf", arch=arch, mode=m, seed=0)
        for src in SOURCES:
            E[f"rq3_loso_{arch}_{src}"] = dict(kind="loso", arch=arch, src=src, seed=0)
        E[f"rq3_code_{arch}"] = dict(kind="code", arch=arch, seed=0)
        for a in ABLATIONS:
            E[f"rq4_{a}_{arch}"] = dict(kind="ablate", arch=arch, abl=a, seed=0)
        for L in [16, 32, 64, 128, 512]:
            E[f"rq4_len{L}_{arch}"] = dict(kind="main", arch=arch, mode="output", seed=0, max_len=L)
    E["rq1_model18_cnn"] = dict(kind="model18", arch="cnn", seed=0)
    E["rq1_model18_lstm"] = dict(kind="model18", arch="lstm", seed=0)
    return E


def run(name, cfg, df):
    arch, seed = cfg["arch"], cfg["seed"]
    epochs = int(os.environ.get(f"EPOCHS_{arch.upper()}", os.environ.get("EPOCHS", 10)))
    tr, va, te = group_split(df, seed=0)  # fixed prompt split across all in-domain experiments
    kind = cfg["kind"]
    if kind == "main":
        tests = {"test": te, **{f"src_{s}": te[te.source == s] for s in SOURCES},
                 "coding": te[te.input.str.contains(CODE_RE)]}
        r, _, _ = run_neural(arch, tr, va, tests, mode=cfg["mode"], max_len=cfg.get("max_len", 256),
                             seed=seed, epochs=epochs)
    elif kind == "model18":
        models = sorted(df.model.unique(), key=lambda m: (FAMILIES.index(df[df.model == m].family.iloc[0]), m))
        r, _, _ = run_neural(arch, tr, va, {"test": te}, seed=seed, epochs=epochs, label_col="model",
                             classes=models)
        r["classes"] = models
    elif kind == "conf":
        d = confounded_dataset(df)
        tr2, va2, te2 = group_split(d, seed=0)
        r, _, _ = run_neural(arch, tr2, va2, {"test": te2}, mode=cfg["mode"], seed=seed, epochs=epochs)
        r["n_rows"] = len(d)
        r["cluster_family_purity"] = float((d.cluster == d.family.map(FAM2ID)).mean())
    elif kind == "loso":
        pool, held = df[df.source != cfg["src"]], df[df.source == cfg["src"]]
        tr2, va2 = split_with_val(pool, seed=0)
        r, _, _ = run_neural(arch, tr2, va2, {"test": held}, seed=seed, epochs=epochs)
        r["n_train"], r["n_test"] = len(tr2), len(held)
    elif kind == "code":
        is_code = df.input.str.contains(CODE_RE)
        pool, held = df[~is_code], df[is_code]
        tr2, va2 = split_with_val(pool, seed=0)
        r, _, _ = run_neural(arch, tr2, va2, {"test": held}, seed=seed, epochs=epochs)
        r["n_train"], r["n_test"], r["n_code_prompts"] = len(tr2), len(held), int(held.prompt_id.nunique())
    elif kind == "ablate":
        f = ABLATIONS[cfg["abl"]]
        tf = lambda d: d.assign(output=d.output.map(f))
        te_t = tf(te)
        r, _, _ = run_neural(arch, tf(tr), tf(va), {"test": te_t}, seed=seed, epochs=epochs)
    r["name"], r["cfg"] = name, cfg
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    E = experiments()
    if a.list:
        print("\n".join(E)); sys.exit()
    df = load_df()
    for n in a.names:
        out = f"{RES}/{n}.json"
        if os.path.exists(out):
            print("skip", n); continue
        print("==>", n, E[n], flush=True)
        r = run(n, E[n], df)
        save_json(r, out)
        print(f"<== {n}: acc={r['test']['test']['acc']:.4f} f1={r['test']['test']['macro_f1']:.4f}", flush=True)
