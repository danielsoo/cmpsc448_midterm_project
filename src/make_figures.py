"""Create report figures and a summary table from results/*.json (skips anything missing)."""
import glob, json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import FAMILIES, load_df, tokenize

R, FIG = "results", "figures"
os.makedirs(FIG, exist_ok=True)
FAM_COL = dict(zip(FAMILIES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
ARCH_COL = {"CNN": "#2a78d6", "BiLSTM": "#eb6834", "TF-IDF+LR": "#8a8984"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "figure.dpi": 200, "savefig.bbox": "tight", "legend.frameon": False})


def J(name):
    p = f"{R}/{name}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def acc(name, key="test", f="acc"):
    r = J(name)
    return None if r is None else r["test"][key][f]


def bar_groups(ax, groups, series, values, colors, ylabel="Accuracy", chance=None, fmt="{:.2f}"):
    n = len(series); w = 0.8 / n; x = np.arange(len(groups))
    for i, s in enumerate(series):
        v = [np.nan if values[s][g] is None else values[s][g] for g in groups]
        bars = ax.bar(x + (i - (n - 1) / 2) * w, v, w * 0.92, color=colors[s], label=s)
        for b, val in zip(bars, v):
            if not np.isnan(val):
                ax.text(b.get_x() + b.get_width() / 2, val + 0.01, fmt.format(val), ha="center", va="bottom",
                        fontsize=6.5, color=MUTED)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    if chance:
        ax.axhline(chance, color=MUTED, ls="--", lw=0.8); ax.text(len(groups) - 0.5, chance + 0.01, "chance",
                                                                  ha="right", fontsize=7, color=MUTED)
    ax.set_ylabel(ylabel); ax.legend(ncol=n, loc="upper left", fontsize=7.5)


def main():
    df = load_df()
    base = json.load(open(f"{R}/baselines.json"))
    summary = {}

    # ---- Fig 1: dataset composition (length distribution per family)
    df["n_tok"] = df.output.map(lambda t: len(tokenize(t)))
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.4), gridspec_kw={"width_ratios": [1, 1.4]})
    cnt = df.groupby("family").size().reindex(FAMILIES)
    ax[0].bar(FAMILIES, cnt.values, color=[FAM_COL[f] for f in FAMILIES], width=0.65)
    for i, v in enumerate(cnt.values):
        ax[0].text(i, v + 30, str(v), ha="center", fontsize=7, color=MUTED)
    ax[0].set_ylabel("# responses"); ax[0].set_ylim(0, 2800); ax[0].tick_params(axis="x", labelsize=7, rotation=30)
    data = [df[df.family == f].n_tok.clip(upper=2000).values for f in FAMILIES]
    bp = ax[1].boxplot(data, tick_labels=FAMILIES, showfliers=False, patch_artist=True, widths=0.55,
                       medianprops=dict(color=INK))
    for p, f in zip(bp["boxes"], FAMILIES):
        p.set_facecolor(FAM_COL[f]); p.set_alpha(0.75); p.set_edgecolor(MUTED)
    ax[1].axhline(256, color=MUTED, ls="--", lw=0.8); ax[1].text(0.55, 275, "max_len = 256", ha="left", fontsize=7,
                                                                 color=MUTED)
    ax[1].set_ylabel("tokens per response"); ax[1].tick_params(axis="x", labelsize=7.5)
    fig.savefig(f"{FIG}/fig1_dataset.png"); plt.close(fig)

    # ---- RQ1 summary over seeds
    rq1 = {}
    for arch, lab in [("cnn", "CNN"), ("lstm", "BiLSTM")]:
        rs = [J(f"rq1_{arch}_s{s}") for s in range(3)]
        rs = [r for r in rs if r]
        if rs:
            a = [r["test"]["test"]["acc"] for r in rs]; f = [r["test"]["test"]["macro_f1"] for r in rs]
            rq1[lab] = dict(acc_mean=float(np.mean(a)), acc_std=float(np.std(a)), f1_mean=float(np.mean(f)),
                            f1_std=float(np.std(f)), n_seeds=len(rs), n_params=rs[0]["n_params"],
                            epochs=[len(r["history"]) for r in rs], train_time=[r["train_time"] for r in rs],
                            per_class_f1=np.mean([r["test"]["test"]["per_class_f1"] for r in rs], 0).tolist())
    rq1["TF-IDF+LR"] = dict(acc_mean=base["tfidf_output"]["test"]["acc"], f1_mean=base["tfidf_output"]["test"]["macro_f1"],
                           per_class_f1=base["tfidf_output"]["test"]["per_class_f1"])
    summary["rq1"] = rq1

    # ---- Fig 2: confusion matrices (seed 0)
    cms = [(lab, J(f"rq1_{a}_s0")) for a, lab in [("cnn", "CNN"), ("lstm", "BiLSTM")]]
    cms = [(l, r) for l, r in cms if r]
    if cms:
        fig, axs = plt.subplots(1, len(cms), figsize=(3.4 * len(cms), 3.0))
        axs = np.atleast_1d(axs)
        for ax, (lab, r) in zip(axs, cms):
            cm = np.array(r["test"]["test"]["confusion"], float); cmn = cm / cm.sum(1, keepdims=True)
            ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
            for i in range(6):
                for j in range(6):
                    ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                            color="white" if cmn[i, j] > 0.5 else INK)
            ax.set_xticks(range(6)); ax.set_xticklabels(FAMILIES, rotation=40, fontsize=7)
            ax.set_yticks(range(6)); ax.set_yticklabels(FAMILIES, fontsize=7)
            ax.set_xlabel("predicted"); ax.set_ylabel("true")
            ax.set_title(f"{lab} (acc {r['test']['test']['acc']:.3f})", fontsize=9)
        fig.tight_layout(); fig.savefig(f"{FIG}/fig2_confusion.png"); plt.close(fig)

    # ---- Fig 3: training curves
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.3))
    for a, lab in [("cnn", "CNN"), ("lstm", "BiLSTM")]:
        r = J(f"rq1_{a}_s0")
        if r:
            h = pd.DataFrame(r["history"])
            ax[0].plot(h.epoch, h.train_loss, marker="o", ms=3, lw=1.6, color=ARCH_COL[lab], label=lab)
            ax[1].plot(h.epoch, h.val_acc, marker="o", ms=3, lw=1.6, color=ARCH_COL[lab], label=lab)
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("train loss"); ax[0].legend()
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("validation accuracy"); ax[1].legend()
    fig.tight_layout(); fig.savefig(f"{FIG}/fig3_training.png"); plt.close(fig)

    # ---- RQ2
    modes = ["input", "output", "both"]
    nat = {"CNN": {}, "BiLSTM": {}, "TF-IDF+LR": {}}
    conf = {"CNN": {}, "BiLSTM": {}, "TF-IDF+LR": {}}
    for a, lab in [("cnn", "CNN"), ("lstm", "BiLSTM")]:
        for m in modes:
            nat[lab][m] = acc(f"rq1_{a}_s0") if m == "output" else acc(f"rq2_{a}_{m}")
            conf[lab][m] = acc(f"rq2conf_{a}_{m}")
    for m in modes:
        nat["TF-IDF+LR"][m] = base[f"tfidf_{m}"]["test"]["acc"]
        conf["TF-IDF+LR"][m] = base[f"conf_tfidf_{m}"]["test"]["acc"]
    summary["rq2"] = dict(natural=nat, confounded=conf)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    lbl = {"input": "Input only", "output": "Output only", "both": "Input + Output"}
    for k, (d, t) in enumerate([(nat, "(a) AlpacaEval (shared prompts)"), (conf, "(b) Simulated biased collection")]):
        dd = {s: {lbl[m]: v for m, v in d[s].items()} for s in d}
        bar_groups(ax[k], list(lbl.values()), list(d), dd, ARCH_COL, chance=1 / 6)
        ax[k].set_title(t, fontsize=9); ax[k].set_ylim(0, 0.8)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig4_rq2.png"); plt.close(fig)

    # ---- RQ3
    srcs = ["helpful_base", "koala", "oasst", "selfinstruct", "vicuna"]
    rq3 = {"in_domain": {}, "cross_domain": {}}
    for a, lab in [("cnn", "CNN"), ("lstm", "BiLSTM")]:
        rq3["in_domain"][lab] = {s: acc(f"rq1_{a}_s0", f"src_{s}") for s in srcs}
        rq3["cross_domain"][lab] = {s: acc(f"rq3_loso_{a}_{s}") for s in srcs}
        rq3["cross_domain"][lab]["coding"] = acc(f"rq3_code_{a}")
    rq3["in_domain"]["TF-IDF+LR"] = {s: base["tfidf_output"][f"src_{s}"]["acc"] for s in srcs}
    rq3["cross_domain"]["TF-IDF+LR"] = {s: base[f"loso_tfidf_{s}"]["test"]["acc"] for s in srcs}
    rq3["cross_domain"]["TF-IDF+LR"]["coding"] = base["code_tfidf"]["test"]["acc"]
    rq3["code_in_domain_tfidf"] = base["code_in_domain_tfidf"]["test"]["acc"]
    rq3["code_per_class"] = {lab: (J(f"rq3_code_{a}") or {}).get("test", {}).get("test", {}).get("per_class_f1")
                             for a, lab in [("cnn", "CNN"), ("lstm", "BiLSTM")]}
    rq3["loso_per_class"] = {lab: {s: (J(f"rq3_loso_{a}_{s}") or {}).get("test", {}).get("test", {}).get("per_class_f1")
                                   for s in srcs} for a, lab in [("cnn", "CNN"), ("lstm", "BiLSTM")]}
    ci = [J(f"rq1_cnn_s{k}")["test"]["coding"]["acc"] for k in (1, 2)
          if J(f"rq1_cnn_s{k}") and "coding" in J(f"rq1_cnn_s{k}")["test"]]
    summary["code_in_domain_cnn"] = float(np.mean(ci)) if ci else None
    rc = J("rq3_code_cnn")
    summary["n_code_prompts"] = rc["n_code_prompts"] if rc else None
    summary["rq3"] = rq3
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for k, lab in enumerate(["CNN", "TF-IDF+LR"]):
        vals = {"in-domain (random prompt split)": rq3["in_domain"][lab],
                "held-out source (LOSO)": {s: rq3["cross_domain"][lab][s] for s in srcs}}
        bar_groups(ax[k], srcs, list(vals), vals,
                   {"in-domain (random prompt split)": "#2a78d6", "held-out source (LOSO)": "#eda100"}, chance=1 / 6)
        ax[k].set_title(lab, fontsize=9); ax[k].set_ylim(0, 0.95); ax[k].tick_params(axis="x", labelsize=7, rotation=20)
        ax[k].legend(ncol=2, loc="upper left", fontsize=6.5)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig5_rq3.png"); plt.close(fig)

    # ---- RQ4 ablations
    abls = ["noformat", "noopener", "funcwords", "contentonly", "shuffle"]
    rq4 = {"ablation": {}, "length": {}}
    for a, lab in [("cnn", "CNN"), ("lstm", "BiLSTM")]:
        rq4["ablation"][lab] = {"original": acc(f"rq1_{a}_s0"), **{x: acc(f"rq4_{x}_{a}") for x in abls}}
        rq4["length"][lab] = {L: acc(f"rq4_len{L}_{a}") for L in [16, 32, 64, 128]}
        rq4["length"][lab][256] = acc(f"rq1_{a}_s0")
    rq4["stylo"] = json.load(open(f"{R}/rq4_stylo.json"))
    summary["rq4"] = {k: v for k, v in rq4.items() if k != "stylo"}
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.6, 1]})
    names = {"original": "original", "noformat": "no\nformatting", "noopener": "no first\nline", "funcwords":
             "function\nwords only", "contentonly": "content\nwords only", "shuffle": "shuffled\nwords"}
    vals = {lab: {names[k]: v for k, v in rq4["ablation"][lab].items()} for lab in ["CNN", "BiLSTM"]}
    bar_groups(ax[0], list(names.values()), ["CNN", "BiLSTM"], vals, ARCH_COL, chance=1 / 6)
    ax[0].set_ylim(0, 0.75); ax[0].tick_params(axis="x", labelsize=6.8); ax[0].set_title("(a) Retrained on ablated text",
                                                                                        fontsize=9)
    for lab in ["CNN", "BiLSTM"]:
        pts = [(L, v) for L, v in rq4["length"][lab].items() if v is not None]
        if len(pts) > 1:
            ax[1].plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=4, lw=1.8, color=ARCH_COL[lab],
                       label=lab)
    ax[1].set_xscale("log", base=2); ax[1].set_xticks([16, 32, 64, 128, 256]); ax[1].set_xticklabels([16, 32, 64, 128, 256])
    ax[1].axhline(1 / 6, color=MUTED, ls="--", lw=0.8); ax[1].set_ylim(0, 0.75)
    ax[1].set_xlabel("tokens seen (prefix length)"); ax[1].set_ylabel("Accuracy"); ax[1].legend()
    ax[1].set_title("(b) How early does the fingerprint appear?", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig6_rq4_ablation.png"); plt.close(fig)

    # ---- RQ4 stylometric heatmap (z-scored per feature)
    st = pd.read_csv(f"{R}/rq4_stats.csv", index_col=0)
    show = ["n_words", "n_paragraphs", "frac_bullet_lines", "frac_numbered_lines", "n_headers", "n_bold",
            "ttr_first100", "avg_sent_len", "excl_per_100w", "comma_per_100w", "closing_offer", "op_certainly",
            "op_sure", "op_here", "op_title"]
    S = st[show]; Z = (S - S.mean()) / S.std()
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.imshow(Z.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto"); ax.grid(False)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            v = S.values[i, j]
            ax.text(j, i, f"{v:.2f}" if v < 10 else f"{v:.0f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(Z.values[i, j]) > 1.3 else INK)
    ax.set_yticks(range(6)); ax.set_yticklabels(Z.index, fontsize=7.5)
    ax.set_xticks(range(len(show))); ax.set_xticklabels(show, rotation=40, ha="right", fontsize=6.8)
    fig.savefig(f"{FIG}/fig7_stylometry.png"); plt.close(fig)

    # ---- model-level (18-way)
    r18 = J("rq1_model18_cnn")
    if r18:
        summary["model18_cnn"] = dict(acc=r18["test"]["test"]["acc"], f1=r18["test"]["test"]["macro_f1"],
                                      classes=r18["classes"], per_class_f1=r18["test"]["test"]["per_class_f1"])
        cm = np.array(r18["test"]["test"]["confusion"], float)
        fam_of = [df[df.model == m].family.iloc[0] for m in r18["classes"]]
        same_fam = sum(cm[i, j] for i in range(18) for j in range(18) if i != j and fam_of[i] == fam_of[j])
        summary["model18_cnn"]["frac_errors_within_family"] = float(same_fam / (cm.sum() - np.trace(cm)))
    json.dump(summary, open(f"{R}/summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1)[:6000])


if __name__ == "__main__":
    main()
