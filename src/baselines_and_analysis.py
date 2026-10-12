"""Non-neural baselines (RQ1/RQ2/RQ3) and interpretable analyses for RQ4.

Outputs (results/):
  baselines.json        TF-IDF + logistic regression for input / output / input+output, confounded data, cross-domain
  rq4_stats.csv         per-family descriptive statistics (length, lexical diversity, formatting)
  rq4_stylo.json        classifier trained ONLY on ~25 hand-crafted stylometric features + feature importance
  rq4_top_ngrams.json   most discriminative n-grams per family (from the TF-IDF logistic regression)
  rq4_openers.json      most frequent opening words per family
"""
import json, os, re, sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from common import FAMILIES, FAM2ID, group_split, load_df, metrics, save_json
from run_experiments import CODE_RE, SOURCES, confounded_dataset, split_with_val

R = "results"


def text_of(d, mode):
    if mode == "input":
        return d.input.tolist()
    if mode == "output":
        return d.output.tolist()
    return (d.input + " [SEP] " + d.output).tolist()


def tfidf_lr(tr, te_dict, mode="output", return_model=False):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100000, sublinear_tf=True,
                          token_pattern=r"(?u)\b\w+\b|[^\w\s]", lowercase=False)
    Xtr = vec.fit_transform(text_of(tr, mode))
    clf = LogisticRegression(C=5.0, max_iter=2000)
    clf.fit(Xtr, tr.family.map(FAM2ID))
    res = {k: metrics(d.family.map(FAM2ID).values, clf.predict(vec.transform(text_of(d, mode))))
           for k, d in te_dict.items()}
    return (res, vec, clf) if return_model else res


# ----------------------------------------------------------------------------- stylometric features
OPENERS = {"op_certainly": r"^(certainly|absolutely|of course)\b", "op_sure": r"^sure\b",
           "op_here": r"^here\b", "op_i": r"^i\b", "op_title": r"^(#|\*\*)"}


def stylo(t):
    words = re.findall(r"[A-Za-z']+", t)
    nw = max(len(words), 1)
    lines = [l for l in t.split("\n") if l.strip()]
    nl = max(len(lines), 1)
    sents = [s for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
    first100 = [w.lower() for w in words[:100]]
    low = t.strip().lower()
    f = dict(
        n_words=len(words), n_chars=len(t), n_lines=len(lines),
        n_paragraphs=len([p for p in re.split(r"\n\s*\n", t) if p.strip()]),
        frac_bullet_lines=sum(bool(re.match(r"\s*[-*+•]\s", l)) for l in lines) / nl,
        frac_numbered_lines=sum(bool(re.match(r"\s*\d+[.)]\s", l)) for l in lines) / nl,
        n_headers=len(re.findall(r"(?m)^\s{0,3}#{1,6}\s", t)),
        n_bold=t.count("**") // 2, n_code_blocks=t.count("```") // 2,
        avg_word_len=np.mean([len(w) for w in words]) if words else 0,
        avg_sent_len=nw / max(len(sents), 1),
        ttr_first100=len(set(first100)) / max(len(first100), 1),
        excl_per_100w=100 * t.count("!") / nw, quest_per_100w=100 * t.count("?") / nw,
        comma_per_100w=100 * t.count(",") / nw, colon_per_100w=100 * t.count(":") / nw,
        upper_ratio=sum(c.isupper() for c in t) / max(len(t), 1),
        digit_ratio=sum(c.isdigit() for c in t) / max(len(t), 1),
        emoji=len(re.findall(r"[\U0001F300-\U0001FAFF☀-➿]", t)),
        ends_with_question=float(low.endswith("?")),
        mentions_ai=float(bool(re.search(r"\bas an ai\b|\blanguage model\b|\bi'm an ai\b", low))),
        closing_offer=float(bool(re.search(r"let me know|hope (this|that) helps|feel free", low))),
        summary_phrase=float(bool(re.search(r"\b(in summary|in conclusion|overall,)", low))),
    )
    for k, p in OPENERS.items():
        f[k] = float(bool(re.match(p, low)))
    return f


def main():
    df = load_df()
    tr, va, te = group_split(df, seed=0)
    trv = pd.concat([tr, va])
    out = {}

    # chance / majority
    out["chance_acc"] = 1 / len(FAMILIES)

    # RQ1 / RQ2: TF-IDF + LR
    for mode in ["input", "output", "both"]:
        tests = {"test": te, **{f"src_{s}": te[te.source == s] for s in SOURCES}}
        out[f"tfidf_{mode}"] = tfidf_lr(trv, tests, mode)
        print(mode, out[f"tfidf_{mode}"]["test"]["acc"], flush=True)

    # RQ2 confounded collection
    d = confounded_dataset(df)
    ctr, cva, cte = group_split(d, seed=0)
    for mode in ["input", "output", "both"]:
        out[f"conf_tfidf_{mode}"] = tfidf_lr(pd.concat([ctr, cva]), {"test": cte}, mode)
        print("conf", mode, out[f"conf_tfidf_{mode}"]["test"]["acc"], flush=True)
    out["conf_rows"] = len(d)

    # RQ3 cross-domain baselines
    for s in SOURCES:
        out[f"loso_tfidf_{s}"] = tfidf_lr(df[df.source != s], {"test": df[df.source == s]})
    is_code = df.input.str.contains(CODE_RE)
    out["code_tfidf"] = tfidf_lr(df[~is_code], {"test": df[is_code]})
    # in-domain reference for coding prompts: random prompt split, restricted to coding prompts in test
    out["code_in_domain_tfidf"] = tfidf_lr(trv, {"test": te[te.input.str.contains(CODE_RE)]})
    save_json(out, f"{R}/baselines.json")

    # ------------------------------------------------------------------ RQ4
    F = pd.DataFrame([stylo(t) for t in df.output])
    F["family"] = df.family.values
    stats = F.groupby("family").mean().reindex(FAMILIES)
    stats.to_csv(f"{R}/rq4_stats.csv")
    print(stats.round(3).T, flush=True)

    feats = [c for c in F.columns if c != "family"]
    # recover row positions of train / test (group_split resets index) via prompt ids
    tr_mask = df.prompt_id.isin(trv.prompt_id).values
    Xtr, Xte = F.loc[tr_mask, feats].values, F.loc[~tr_mask, feats].values
    ytr, yte = df.family.map(FAM2ID).values[tr_mask], df.family.map(FAM2ID).values[~tr_mask]
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(Xtr, ytr)
    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=0).fit(Xtr, ytr)
    pi = permutation_importance(gb, Xte, yte, n_repeats=5, random_state=0, scoring="accuracy")
    imp = sorted(zip(feats, pi.importances_mean.tolist()), key=lambda x: -x[1])
    sty = dict(features=feats, lr=metrics(yte, lr.predict(Xte)), gb=metrics(yte, gb.predict(Xte)),
               gb_perm_importance=imp)
    # structural-only subset vs lexical subset
    groups = {"length": ["n_words", "n_chars", "n_lines", "n_paragraphs", "avg_sent_len"],
              "formatting": ["frac_bullet_lines", "frac_numbered_lines", "n_headers", "n_bold", "n_code_blocks",
                             "colon_per_100w", "op_title"],
              "lexical_phrasal": ["avg_word_len", "ttr_first100", "excl_per_100w", "quest_per_100w", "comma_per_100w",
                                  "upper_ratio", "digit_ratio", "emoji", "ends_with_question", "mentions_ai",
                                  "closing_offer", "summary_phrase", "op_certainly", "op_sure", "op_here", "op_i"]}
    sty["groups"] = {}
    for g, cols in groups.items():
        idx = [feats.index(c) for c in cols]
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=0).fit(Xtr[:, idx], ytr)
        sty["groups"][g] = metrics(yte, m.predict(Xte[:, idx]))
    save_json(sty, f"{R}/rq4_stylo.json")
    print("stylo LR", sty["lr"]["acc"], "GB", sty["gb"]["acc"], {g: v["acc"] for g, v in sty["groups"].items()})

    # top n-grams per family (output-only TF-IDF LR trained on train+val)
    _, vec, clf = tfidf_lr(trv, {"test": te}, "output", return_model=True)
    names = np.array(vec.get_feature_names_out())
    top = {}
    for i, f in enumerate(FAMILIES):
        top[f] = names[np.argsort(-clf.coef_[i])[:25]].tolist()
    save_json(top, f"{R}/rq4_top_ngrams.json")

    # opening words
    op = {}
    for f in FAMILIES:
        firsts = df[df.family == f].output.map(lambda t: " ".join(re.findall(r"\w+|[^\w\s]", t)[:2]))
        op[f] = [(k, round(v / len(firsts), 3)) for k, v in Counter(firsts).most_common(8)]
    save_json(op, f"{R}/rq4_openers.json")


if __name__ == "__main__":
    main()
