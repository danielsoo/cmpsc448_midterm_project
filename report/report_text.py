# Report text. Executed inside build_report.py (S, B, STY, TOP, OPN, J, pct, fig, img available).


def rq1_table():
    r = S["rq1"]
    rows = ""
    for lab in ["CNN", "BiLSTM"]:
        if lab not in r:
            continue
        x = r[lab]
        sd = lambda v: f" &plusmn; {100 * v:.1f}" if x["n_seeds"] > 1 else ""
        rows += (f"<tr><td class='l'>{lab} (ours)</td><td>{pct(x['acc_mean'])}{sd(x['acc_std'])}</td>"
                 f"<td>{pct(x['f1_mean'])}{sd(x['f1_std'])}</td><td>{x['n_seeds']}</td>"
                 f"<td>{x['n_params'] / 1e6:.2f}M</td></tr>")
    t = r["TF-IDF+LR"]
    rows += (f"<tr><td class='l'>TF-IDF (1-2 gram) + logistic regression</td><td>{pct(t['acc_mean'])}</td>"
             f"<td>{pct(t['f1_mean'])}</td><td>-</td><td>-</td></tr>")
    rows += (f"<tr><td class='l'>Stylometric features only (gradient boosting)</td><td>{pct(STY['gb']['acc'])}</td>"
             f"<td>{pct(STY['gb']['macro_f1'])}</td><td>-</td><td>{len(STY['features'])} feats</td></tr>")
    rows += "<tr><td class='l'>Chance (6 balanced classes)</td><td>16.7</td><td>-</td><td>-</td><td>-</td></tr>"
    return ("<table><caption><b>Table 3.</b> RQ1: family identification from the response only "
            "(test set: 2,896 responses to 161 unseen prompts). Mean &plusmn; std over seeds.</caption>"
            "<tr><th class='l'>Model</th><th>Accuracy (%)</th><th>Macro-F1 (%)</th><th>Seeds</th><th>Params</th></tr>"
            + rows + "</table>")


def per_class_table():
    r = S["rq1"]
    hdr = "".join(f"<th>{f}</th>" for f in FAMILIES)
    rows = ""
    for lab in ["CNN", "BiLSTM", "TF-IDF+LR"]:
        if lab in r:
            rows += f"<tr><td class='l'>{lab}</td>" + "".join(f"<td>{pct(v)}</td>" for v in r[lab]["per_class_f1"]) + "</tr>"
    return ("<table><caption><b>Table 4.</b> Per-family F1 (%) on the RQ1 test set (CNN/BiLSTM averaged over seeds)."
            f"</caption><tr><th class='l'>Model</th>{hdr}</tr>{rows}</table>")


def rq2_table():
    n, c = S["rq2"]["natural"], S["rq2"]["confounded"]
    rows = ""
    for lab in ["CNN", "BiLSTM", "TF-IDF+LR"]:
        rows += (f"<tr><td class='l'>{lab}</td>" + "".join(f"<td>{pct(n[lab][m])}</td>" for m in ["input", "output", "both"])
                 + "".join(f"<td>{pct(c[lab][m])}</td>" for m in ["input", "output", "both"]) + "</tr>")
    return ("<table><caption><b>Table 5.</b> RQ2: accuracy (%) with input only, output only, and input + output. "
            "All runs use seed 0. Left: our AlpacaEval dataset. Right: a simulated biased collection (Section 5.2). Chance = 16.7.</caption>"
            "<tr><th rowspan=2 class='l'>Model</th><th colspan=3>AlpacaEval (shared prompts)</th>"
            "<th colspan=3>Biased collection</th></tr><tr><th>Input</th><th>Output</th><th>In+Out</th>"
            "<th>Input</th><th>Output</th><th>In+Out</th></tr>" + rows + "</table>")


def rq3_table():
    r = S["rq3"]
    srcs = ["helpful_base", "koala", "oasst", "selfinstruct", "vicuna"]
    rows = ""
    code_in = {"CNN": S.get("code_in_domain_cnn"), "BiLSTM": None, "TF-IDF+LR": r.get("code_in_domain_tfidf")}
    for lab in ["CNN", "BiLSTM", "TF-IDF+LR"]:
        ind = r["in_domain"][lab]; cro = r["cross_domain"][lab]
        if all(cro.get(s) is None for s in srcs) and cro.get("coding") is None:
            continue
        rows += f"<tr><td class='l'>{lab}</td>" + "".join(
            f"<td>{pct(ind.get(s))} &rarr; {pct(cro.get(s))}</td>" for s in srcs) + f"<td>{pct(code_in.get(lab))} &rarr; {pct(cro.get('coding'))}</td></tr>"
    return ("<table><caption><b>Table 6.</b> RQ3: accuracy (%) per prompt source. Each cell is "
            "<i>in-domain</i> (test prompts of that source, model trained on all sources) &rarr; <i>cross-domain</i> "
            "(model trained on the other four sources only). Last column: coding prompts; in-domain = the 17 coding prompts (306 responses) in the RQ1 test split (CNN: mean of seeds 1-2), "
            "cross-domain = trained on non-coding prompts only and tested on all coding prompts.</caption><tr><th class='l'>Model</th>" + "".join(f"<th>{s}</th>" for s in srcs)
            + "<th>coding (held out)</th></tr>" + rows + "</table>")


def ngram_table():
    rows = ""
    for f in FAMILIES:
        ng = ", ".join(f"<code>{x}</code>" for x in TOP[f][:10])
        op = ", ".join(f"<code>{a}</code> ({100 * b:.0f}%)" for a, b in OPN[f][:3])
        rows += f"<tr><td class='l'><b>{f}</b></td><td class='l'>{op}</td><td class='l'>{ng}</td></tr>"
    return ("<table class='ex'><caption><b>Table 7.</b> RQ4: most frequent opening tokens and the n-grams with the "
            "largest positive logistic-regression weight for each family (output-only TF-IDF model).</caption>"
            "<tr><th class='l'>Family</th><th class='l'>Top openings (share)</th><th class='l'>Most indicative n-grams</th></tr>"
            + rows + "</table>")


def HTML_BODY():
    return "".join(SECTIONS())


EXAMPLE = [
    ("gpt4_1106_preview (GPT)", "Tahu Gejrot Cirebon is a popular Indonesian street food originating from Cirebon, a port city in West Java, Indonesia. It's a sweet, spicy, and sour dish made w..."),
    ("claude-3-opus (Claude)", "Certainly! Here's a recipe for Tahu Gejrot Cirebon, a popular Indonesian dish from the city of Cirebon in West Java:<br><br>Ingredients:<br>- 500g fried tofu, cut into b..."),
    ("Llama-3-70B-Instruct (Llama)", "What a great choice! Tahu Gejrot Cirebon is a popular Indonesian street food originating from Cirebon, West Java. It's a delicious and refreshing dish that's pe..."),
    ("gemini-pro (Google)", "**Tahu Gejrot Cirebon**<br><br>**Ingredients:**<br><br>* 20 pieces of fried tofu<br>* 100 grams of red chilies, seeded and chopped<br>..."),
]


def sec_front():
    return f"""
<h1>Who Wrote It? Identifying LLMs from Their Responses</h1>
<div class="sub">CMPSC 448 Midterm Project &middot; Fall 2026</div>
<div class="sub"><b>Team leader (individual project):</b> Younsoo Park</div>
<div class="sub small">Code, data-building scripts and all raw results: see the GitHub repository submitted with this report
(<code>README.md</code> explains how to reproduce every number).</div>
<div class="abstract"><b>Abstract.</b> {ABSTRACT}</div>
"""


def sec_data():
    ex = "".join(f"<tr><td style='width:22%'><b>{m}</b></td><td>{t}</td></tr>" for m, t in EXAMPLE)
    return f"""
<h2>1. Problem Definition</h2>
<p>Given a piece of text produced by a large language model, we want to predict <i>which</i> LLM family produced it.
We cast this as 6-way single-label text classification over the families <b>GPT</b> (OpenAI), <b>Claude</b> (Anthropic),
<b>Llama</b> (Meta), <b>Qwen</b> (Alibaba), <b>Mistral</b> (Mistral AI) and <b>Google</b> (Gemini and Gemma). Each example is a
triple <code>(LLM_name, LLM_input, LLM_output)</code>. The classifier sees the output (RQ1), the input, or both (RQ2), and must
output the family. We report accuracy and macro-F1; with balanced classes, chance accuracy is 1/6 = 16.7%.</p>
<p>Two design principles guided the whole project. (1) <b>No prompt leakage.</b> Because many models answer the same prompt,
a random row-level split would put the same question in train and test, and a model could memorise topic words instead of
style. All our splits are made <i>by prompt</i>, so every test prompt is unseen. (2) <b>Control the prompt.</b> If different
LLMs were queried with different prompts, a classifier could succeed by recognising the prompt distribution rather than the
model. We therefore chose a dataset in which every model answered <i>exactly the same prompts</i>, which makes RQ2 a clean test.</p>

<h2>2. Dataset Curation</h2>
<p><b>Source.</b> We use the public model outputs released with the AlpacaEval benchmark [1]
(<code>github.com/tatsu-lab/alpaca_eval</code>, files <code>results/&lt;model&gt;/model_outputs.json</code>, commit
<code>cd543a1</code>). AlpacaEval asks every model to answer the same 805 instructions, which were pooled from five public
instruction sets: <i>helpful_base</i> (129 prompts), <i>koala</i> (156), <i>oasst</i> (188), <i>selfinstruct</i> (252) and
<i>vicuna</i> (80). Each JSON record already contains the instruction, the output and the generator name, so it maps directly
to <code>(LLM_name, LLM_input, LLM_output)</code>; we additionally keep the instruction source, which we use for RQ3.
The outputs were produced by the benchmark maintainers or model submitters through the official APIs or the released model
weights. The prompts are public benchmark instructions, so the data contains no private or personal information.</p>
<p><b>Selection and balancing.</b> To capture a family-level fingerprint rather than one checkpoint, we picked three
official (non-community-fine-tuned) models per family, spanning different sizes and release dates (Table 1). Empty responses
were removed (24 rows: 23 from Gemma/Gemini, 1 missing Llama-2 output), giving <b>14,466 examples</b>; the six classes contain
2,392 to 2,415 examples each, so the data is balanced by construction. The script <code>src/build_dataset.py</code> rebuilds the
CSV from the raw files.</p>
<table><caption><b>Table 1.</b> Models used (18 models, 6 families, 805 shared prompts each).</caption>
<tr><th class='l'>Family</th><th class='l'>Models</th><th>Responses</th><th>Mean words</th></tr>
{FAMILY_ROWS}
</table>
<p><b>Splits.</b> We split the 805 prompts (not the rows) into 70% train / 10% validation / 20% test with a fixed seed: 563 / 81 / 161
prompts, i.e. 10,114 / 1,456 / 2,896 responses. Because every prompt is answered by all 18 models, each split is also balanced
across families, and the test prompts were never seen during training or model selection. Figure 1 shows the class sizes and the
length distribution. Table 2 shows how differently the families answer one test prompt.</p>
{fig("figures/fig1_dataset.png", "Left: responses per family. Right: response length in tokens (our tokenizer; whiskers at 1.5 IQR). The dashed line is the 256-token input length used by the neural models.", 1)}
<table class='ex'><caption><b>Table 2.</b> Beginnings of four responses to the same prompt (a request for an Indonesian recipe).
The opening phrase and the formatting already differ by family.</caption>{ex}</table>
"""


def sec_models():
    r = S["rq1"]
    cnn_ep = r.get("CNN", {}).get("epochs", ["?"]); lstm_ep = r.get("BiLSTM", {}).get("epochs", ["?"])
    cnn_t = r.get("CNN", {}).get("train_time", [0]); lstm_t = r.get("BiLSTM", {}).get("train_time", [0])
    return f"""
<h2>3. Models and Training</h2>
<p><b>Tokenisation and input representation.</b> Formatting is likely part of an LLM's style, so we did not use a standard
lower-casing word tokenizer. Our regex tokenizer (<code>src/common.py</code>) keeps letter case, splits every punctuation and
markdown symbol into its own token (<code>*</code>, <code>#</code>, <code>-</code>, <code>:</code> ...) and turns every line break
into a special <code>&lt;nl&gt;</code> token. The vocabulary is built on the training split only (tokens with frequency &ge; 2,
capped at 30,000; about {VOCAB} types; unseen tokens map to <code>&lt;unk&gt;</code>). Each response is truncated to its
first 256 tokens (the median response is about 310 tokens and 62% of responses are longer, Figure 1); we study the effect of this length in RQ4. For
input + output (RQ2), the sequence is <code>[input tokens (&le;64)] &lt;sep&gt; [output tokens]</code>, truncated to 256 tokens.</p>
<p><b>CNN.</b> A Kim-style text CNN [2]: 128-d embeddings learned from scratch (embedding dropout 0.2) &rarr; three parallel 1-D
convolutions with kernel widths 3, 4 and 5 and 128 filters each, ReLU &rarr; global max-pooling over time (padding positions are
masked) &rarr; dropout 0.5 &rarr; linear layer to 6 logits. The kernels act as learned n-gram detectors (for example
"<code>Certainly ! Here</code>" or "<code>&lt;nl&gt; * *</code>"), and max-pooling makes the detection position-independent.
Total: {r.get('CNN', {}).get('n_params', 0) / 1e6:.2f}M parameters (most of them in the embedding table).</p>
<p><b>RNN (BiLSTM).</b> 128-d embeddings (dropout 0.2) &rarr; one bidirectional LSTM layer with 64 hidden units per direction,
run on packed sequences so padding is skipped &rarr; concatenation of mean-pooling and max-pooling of the hidden states over time
(256-d) &rarr; dropout 0.4 &rarr; linear layer. We pool over all time steps instead of using only the final state because the fingerprint (an opener, a closing offer) can sit anywhere in the response. Total: {r.get('BiLSTM', {}).get('n_params', 0) / 1e6:.2f}M parameters.
We used 64 hidden units instead of 128 because a 128-unit LSTM took about 3.6 minutes per epoch on our 2-core CPU, which made the
required number of runs infeasible; the 64-unit model takes about 1.8 minutes.</p>
<p><b>Training.</b> Both models are trained with cross-entropy loss, AdamW (weight decay 1e-4; learning rate 1e-3 for the CNN
and 2e-3 for the BiLSTM), batch size 64, and gradient-norm clipping at 1.0. We select the checkpoint with the best validation
macro-F1 and stop early after 3 epochs without improvement (maximum 12 epochs for the CNN; 8 for the main BiLSTM run and 6 for the other BiLSTM runs). With seed 0
the CNN trained for {cnn_ep[0]} epochs ({cnn_t[0] / 60:.1f} min) and the BiLSTM for {lstm_ep[0]} epochs ({lstm_t[0] / 60:.1f} min) on CPU.
Figure 3 shows the learning curves. Our first CNN configuration (learning rate 2e-3 and dropout 0.5 applied directly to the
embeddings) was unstable and stopped at 43% accuracy; lowering the embedding dropout and the learning rate fixed this.
All other experiments reuse these exact settings, and all experiments use the same prompt split.</p>
<p><b>Baselines.</b> To put the neural numbers in context we also train (i) a TF-IDF (word 1-2 grams, case-sensitive,
punctuation kept) + logistic regression classifier and (ii) a gradient-boosting classifier on {len(STY['features'])} hand-crafted
stylometric features (Section 7).</p>
{fig("figures/fig3_training.png", "Training loss and validation accuracy per epoch (seed 0). The selected checkpoint is the epoch with the best validation macro-F1.", 3, "88%")}
"""


import pandas as _pd
_df = _pd.read_csv("data/llm_dataset.csv", keep_default_na=False)
_st = _pd.read_csv("results/rq4_stats.csv", index_col=0)
FAMILY_ROWS = ""
for _f in FAMILIES:
    _ms = sorted(_df[_df.family == _f].model.unique())
    FAMILY_ROWS += (f"<tr><td class='l'><b>{_f}</b></td><td class='l'>{', '.join(_ms)}</td>"
                    f"<td>{(_df.family == _f).sum():,}</td><td>{_st.loc[_f, 'n_words']:.0f}</td></tr>")
VOCAB = f"{J('rq1_cnn_s0')['vocab']:,}"


def sec_refs():
    return """
<h2>References</h2>
<p class="small">[1] X. Li, T. Zhang, Y. Dubois, R. Taori, I. Gulrajani, C. Guestrin, P. Liang, T. B. Hashimoto.
AlpacaEval: An Automatic Evaluator of Instruction-following Models. GitHub repository, 2023. https://github.com/tatsu-lab/alpaca_eval<br>
[2] Y. Kim. Convolutional Neural Networks for Sentence Classification. EMNLP 2014.<br>
[3] S. Hochreiter and J. Schmidhuber. Long Short-Term Memory. Neural Computation, 9(8), 1997.<br>
[4] F. Pedregosa et al. Scikit-learn: Machine Learning in Python. JMLR 12, 2011.<br>
[5] A. Paszke et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS 2019.</p>
<p class="small"><b>Use of AI tools.</b> As permitted by the project description, an LLM assistant was used to help write and
debug code and to draft text. All experiments were run, checked and interpreted by the author, who is responsible for the content.</p>
"""


def _cm_stats(name):
    import numpy as np
    r = J(name)
    if not r:
        return None
    cm = np.array(r["test"]["test"]["confusion"], float)
    recall = np.diag(cm) / cm.sum(1)
    predshare = cm.sum(0) / cm.sum()
    return recall, predshare, cm


def sec_rq1():
    r = S["rq1"]; c, l, t = r.get("CNN", {}), r.get("BiLSTM", {}), r["TF-IDF+LR"]
    rc, pc, cmc = _cm_stats("rq1_cnn_s0"); rl, pl, cml = _cm_stats("rq1_lstm_s0")
    m18 = S.get("model18_cnn")
    def _rng(idx):
        v = [r[m]["per_class_f1"][i] for m in ["CNN", "BiLSTM", "TF-IDF+LR"] if m in r for i in idx]
        return f"{min(v):.2f} and {max(v):.2f}"
    m18txt = ""
    if m18:
        m18txt = (f"<p><b>Model-level (18-way) identification.</b> As an extra test we trained the same CNN to predict the exact "
                  f"model among all 18. It reaches {pct(m18['acc'])}% accuracy (chance 5.6%), and "
                  f"{pct(m18['frac_errors_within_family'], 0)}% of its mistakes confuse two models of the <i>same</i> family "
                  f"(a random error would hit the same family only 2/17 = 12% of the time). So sibling models really do share a style, "
                  f"which supports treating the family as the label.</p>")
    return f"""
<h2>4. RQ1: Can we identify which LLM generated a response?</h2>
{rq1_table()}
<p><b>Yes, well above chance.</b> From the response alone the CNN reaches {pct(c.get('acc_mean'))}% accuracy and the BiLSTM
{pct(l.get('acc_mean'))}%, about 3.2 times chance (16.7%), on prompts never seen in training (Table 3). The two architectures are
close; the CNN is slightly better and about {l.get('train_time', [1])[0] / max(c.get('train_time', [1])[0], 1):.0f} times faster to train
on CPU. The CNN's standard deviation over 3 seeds ({pct(c.get('acc_std'))} points) is larger than the CNN/BiLSTM gap, so we do not claim
one architecture is clearly better (the BiLSTM was run with {l.get('n_seeds')} seed{'s' if l.get('n_seeds', 1) > 1 else ''} because of its training cost). A plain TF-IDF + logistic-regression model is stronger ({pct(t['acc_mean'])}%). The linear model sees the <i>whole</i> response while our networks see only the first 256 tokens, but Section 7.3 shows that the CNN is already saturated at 128 tokens, so truncation is not the main reason. More likely, with only ~10k training texts, embeddings and filters learned from scratch are data-hungry, whereas sparse n-gram counts with a convex classifier are a very strong and sample-efficient model of style.</p>
{fig("figures/fig2_confusion.png", "Row-normalised confusion matrices on the test set (seed 0). Diagonal = recall per family.", 2, "92%")}
<p><b>Which families are easy?</b> Table 4 and Figure 2 show a clear split. <b>Claude, Llama and Google</b> are identified with
F1 between {_rng([1, 2, 5])} by all three classifiers, while <b>GPT and Qwen</b> are the hardest (F1 between {_rng([0, 3])}) and are mostly
confused with each other and with Mistral. For example, the BiLSTM labels {pct(cml[0][3] / cml[0].sum(), 0)}% of GPT responses as Qwen. A plausible
explanation (which we cannot verify from this data) is that GPT-style chat answers were widely used to train open models, so Qwen and Mistral answers look GPT-like. Section 7 also shows that Qwen and Mistral have no feature that stands out strongly, and GPT's most distinctive habits (markdown headers, "Certainly!") occur only in a minority of its answers, while Claude (almost no markdown bold, "Here is/are" openers), Llama ("What a great ...!", many exclamation
marks) and Google (bulleted <code>*</code> lists, a bold title as the first line) each have a distinctive signature. The CNN also
over-predicts Mistral ({pct(pc[4], 0)}% of its predictions vs. 16.7% true share), i.e. Mistral acts as its "default" class for
generic-looking answers, which hurts GPT and Qwen recall.</p>
{per_class_table()}
{m18txt}
"""


def sec_rq2():
    n, c = S["rq2"]["natural"], S["rq2"]["confounded"]
    conf = J("rq2conf_cnn_input") or {}
    return f"""
<h2>5. RQ2: Does the user's prompt help identify the LLM?</h2>
<h3>5.1 Input only, output only, input + output</h3>
{rq2_table()}
{fig("figures/fig4_rq2.png", "RQ2 accuracy. (a) On AlpacaEval every model answered the same prompts, so the input alone carries no information. (b) When each family is queried mostly with prompts of one topic (simulated biased collection), the input alone becomes predictive.", 4)}
<p><b>i) Can the LLM be predicted from the prompt alone?</b> Not on our dataset: every input-only classifier sits at exactly
chance ({pct(n['CNN']['input'])}% CNN, {pct(n['BiLSTM']['input'])}% BiLSTM, {pct(n['TF-IDF+LR']['input'])}% TF-IDF). This is
the expected outcome of our design: each of the 805 prompts appears once per model, i.e. three times with each family label, so
the conditional distribution of the label given the prompt is uniform and no classifier can do better than guessing (the models
collapse to predicting mostly two or three classes, hence the low macro-F1).</p>
<p><b>ii) Does adding the input help?</b> Only a little. For the CNN, input + output gives {pct(n['CNN']['both'])}% vs.
{pct(n['CNN']['output'])}% for output only; for the BiLSTM {pct(n['BiLSTM']['both'])}% vs. {pct(n['BiLSTM']['output'])}%; for TF-IDF
the two are equal ({pct(n['TF-IDF+LR']['both'])}% vs. {pct(n['TF-IDF+LR']['output'])}%). Since the prompt itself is uninformative
here, any gain must come from <i>interactions</i>: the prompt tells the model what kind of answer to expect (a list request, a
coding question, a short factual question), so it can judge whether the formatting or length of the answer is unusual for that
request. The effect is small and comparable to seed-to-seed variation, so we treat it as suggestive rather than conclusive. Note also
that adding up to 64 prompt tokens leaves fewer of the 256 tokens for the response.</p>
<h3>5.2 iii) When would input-only classification work "surprisingly well"?</h3>
<p>Many public "which-LLM" datasets are collected differently from ours: each LLM is queried by different people, at different
times, or with different task lists. Then the prompt distribution itself differs by label, and an input-only classifier can succeed
without ever seeing an LLM's text. To demonstrate this, we built a <b>simulated biased collection</b>: we clustered the 805 prompts
into 6 equal-size topic clusters (TF-IDF + k-means with balanced assignment) and paired each cluster with one family; a
(prompt, family) response is kept with probability 0.9 if they are paired and 0.15 otherwise. This gives {conf.get('n_rows', 4091):,}
responses in which each family is still answered by real LLM text, but {pct(conf.get('cluster_family_purity', 0.54), 0)}% of a
family's examples come from "its" topic. With exactly the same pipeline and prompt-grouped split (test: 844 responses), the input-only
models now beat chance clearly: TF-IDF {pct(c['TF-IDF+LR']['input'])}%, CNN {pct(c['CNN']['input'])}%
and BiLSTM {pct(c['BiLSTM']['input'])}%. These classifiers have learned "who was asked about what", not how the LLMs write.
Output-only accuracy is lower than on the full dataset because there is about 3.5 times less training data. Adding the input helps the CNN ({pct(c['CNN']['both'])}% vs. {pct(c['CNN']['output'])}%) and TF-IDF ({pct(c['TF-IDF+LR']['both'])}% vs. {pct(c['TF-IDF+LR']['output'])}%) here, but not the BiLSTM ({pct(c['BiLSTM']['both'])}% vs. {pct(c['BiLSTM']['output'])}%), although its validation accuracy was higher with the input (48.0% vs. 45.1%). With only 844 test responses, differences of 1 to 2 points are within noise (one standard error is about 1.7 points). <b>Lesson:</b> when input-only works
well, it signals a confound in data collection (topic, time period, user population or language), and output-based results on
such data are inflated by the same confound because the response also mentions the topic. A shared-prompt design like ours removes
it.</p>
"""


def sec_rq4():
    a = S["rq4"]["ablation"]; L = S["rq4"]["length"]
    g = STY["groups"]; imp = STY["gb_perm_importance"][:6]
    st = _st
    def ab(lab, k):
        return pct(a[lab].get(k))
    def drop(lab, k):
        o, v = a[lab].get("original"), a[lab].get(k)
        return "n/a" if o is None or v is None else f"{100 * (v - o):+.1f}"
    lstm_has = any(a["BiLSTM"].get(k) is not None for k in ["noformat", "funcwords", "shuffle"])
    rows = ""
    labels = [("original", "Original text"), ("noformat", "Formatting removed (markdown symbols, list markers, line breaks)"),
              ("noopener", "First line / first sentence removed"),
              ("funcwords", "Function words + punctuation only (content words &rarr; <code>W</code>)"),
              ("contentonly", "Content words only (lower-cased, no stop words, no punctuation, no line breaks)"),
              ("shuffle", "Word order shuffled within each response")]
    for k, name in labels:
        rows += (f"<tr><td class='l'>{name}</td><td>{ab('CNN', k)}</td><td>{'' if k == 'original' else drop('CNN', k)}</td>"
                 + (f"<td>{ab('BiLSTM', k)}</td><td>{'' if k == 'original' else drop('BiLSTM', k)}</td>" if lstm_has else "")
                 + "</tr>")
    hdr = "<th>CNN acc</th><th>&Delta;</th>" + ("<th>BiLSTM acc</th><th>&Delta;</th>" if lstm_has else "")
    abl_table = (f"<table><caption><b>Table 8.</b> RQ4 ablations: the text is transformed in train, validation and test, and the model "
                 f"is retrained from scratch (accuracy %, &Delta; = change vs. original).{'' if lstm_has else ' BiLSTM ablations were not run (compute budget).'}"
                 f"</caption><tr><th class='l'>Input text</th>{hdr}</tr>{rows}</table>")
    imp_txt = ", ".join(f"<code>{n}</code> ({100 * v:.1f})" for n, v in imp)
    return f"""
<h2>7. RQ4: What characteristics distinguish different LLMs?</h2>
<h3>7.1 Descriptive statistics</h3>
<p>We computed {len(STY['features'])} interpretable features per response (length, paragraphs, list and header usage, markdown bold,
code blocks, sentence and word length, type-token ratio of the first 100 words, punctuation rates, and opening/closing phrases).
Figure 7 shows the family means. The families differ in clear and readable ways:</p>
<ul>
<li><b>Length:</b> Llama ({st.loc['Llama', 'n_words']:.0f} words on average) and GPT ({st.loc['GPT', 'n_words']:.0f}) write the longest
answers, Claude ({st.loc['Claude', 'n_words']:.0f}) and Google ({st.loc['Google', 'n_words']:.0f}) the shortest.</li>
<li><b>Markdown bold:</b> Claude essentially never uses <code>**bold**</code> ({st.loc['Claude', 'n_bold']:.3f} per response) while
Google, Llama, GPT and Qwen use about 3 to 5 per response. Mistral uses little ({st.loc['Mistral', 'n_bold']:.2f}).</li>
<li><b>Lists and headers:</b> Google prefers bullet lists ({100 * st.loc['Google', 'frac_bullet_lines']:.0f}% of lines) over numbered
lists; GPT is the only family that regularly uses markdown headers (<code>###</code>; {st.loc['GPT', 'n_headers']:.2f} per response).</li>
<li><b>Openers:</b> {100 * st.loc['Claude', 'op_here']:.0f}% of Claude answers start with "Here ...", {100 * st.loc['GPT', 'op_certainly']:.0f}% of
GPT answers with "Certainly/Absolutely/Of course", {100 * st.loc['Google', 'op_title']:.0f}% of Google answers with a bold title or header,
and Llama has the most exclamation marks ("What a great question!") (Table 7).</li>
<li><b>Lexical diversity</b> differs only slightly (type-token ratio of the first 100 words: {st['ttr_first100'].min():.2f} to
{st['ttr_first100'].max():.2f}).</li>
</ul>
{fig("figures/fig7_stylometry.png", "Mean value of selected stylometric features per family (cell colour = z-score across families; red = above average, blue = below).", 7)}
{ngram_table()}
<h3>7.2 Which signals does a classifier use?</h3>
<p>A gradient-boosting classifier trained <i>only</i> on the {len(STY['features'])} hand-crafted features reaches
{pct(STY['gb']['acc'])}% accuracy, already above our neural models and close to the TF-IDF model. Using one feature group at a time:
formatting features alone give {pct(g['formatting']['acc'])}%, length features alone {pct(g['length']['acc'])}%, and lexical/phrasal
features alone {pct(g['lexical_phrasal']['acc'])}%. The most important features by permutation importance (accuracy drop in points
when the feature is shuffled) are {imp_txt}. So the fingerprint is <b>both structural and linguistic</b>, but structure (formatting
and length) is the single strongest group. The TF-IDF n-grams in Table 7 tell the same story: the top weights are markdown
patterns (<code># #</code>, <code>* *</code>, <code>: -</code>) and stock phrases ("Certainly !", "Here is", "I hope", "Remember ,",
"happy to help", "The answer is").</p>
<h3>7.3 Does removing signals reduce performance? (ablations)</h3>
{abl_table}
{fig("figures/fig6_rq4_ablation.png", "(a) Accuracy after retraining on transformed text. (b) Accuracy when the model only sees the first k tokens of each response (CNN retrained for each k).", 6)}
{ABLATION_TEXT}
"""


def sec_rq3():
    r = S["rq3"]; srcs = ["helpful_base", "koala", "oasst", "selfinstruct", "vicuna"]
    def avg(d):
        v = [d.get(s) for s in srcs if d.get(s) is not None]
        return sum(v) / len(v) if v else None
    rows_avg = {lab: (avg(r["in_domain"][lab]), avg(r["cross_domain"][lab])) for lab in ["CNN", "BiLSTM", "TF-IDF+LR"]}
    code_cnn_in = S.get("code_in_domain_cnn")
    return f"""
<h2>6. RQ3: Do LLM fingerprints generalise across tasks or domains?</h2>
<p><b>Setup.</b> We use two kinds of domain shift. (a) <b>Leave-one-source-out (LOSO):</b> the 805 prompts come from five instruction
sets with different styles (e.g. <i>selfinstruct</i> holds short, task-like instructions such as "Rewrite this sentence ..."; <i>vicuna</i>
and <i>koala</i> hold open-ended chat questions; <i>oasst</i> and <i>helpful_base</i> come from real user conversations). For each source
we train on the other four (with a 12% prompt-grouped validation split) and test on all responses from the held-out source.
(b) <b>General &rarr; coding:</b> we mark the {S.get('n_code_prompts', 79)} prompts that ask for code (regex on words such as
<i>code, python, function, SQL, script, algorithm</i>), train on all non-coding prompts and test on the coding prompts. This mirrors the
"train on general QA, test on coding" example in the project description. The in-domain reference is the RQ1 model evaluated on the test
prompts of the same source (or on the coding prompts of the RQ1 test split).</p>
{rq3_table()}
{fig("figures/fig5_rq3.png", "RQ3 in-domain vs. held-out-source accuracy per source (CNN and TF-IDF baseline).", 5)}
{RQ3_TEXT}
"""


def sec_lessons():
    return f"""
<h2>8. Lessons and Experience</h2>
<ul>
<li><b>The data design decides what a result means.</b> Our most important decision was not an architecture but the dataset:
using shared prompts and prompt-grouped splits. Section 5.2 shows that with a biased collection, a classifier can look clever
while only recognising the topic. Before trusting any "which-LLM" accuracy we would now always ask: can the input alone predict the
label? If yes, the dataset has a confound.</li>
<li><b>Prompt leakage is easy to miss.</b> Every prompt appears 18 times in our data. A random row split would have put the same
question (and therefore the same topic words) in train and test. Grouping by prompt is a one-line change
(<code>GroupShuffleSplit</code>) but it is essential for an honest test set.</li>
<li><b>Strong simple baselines are humbling and useful.</b> TF-IDF + logistic regression beat both of our neural networks, and {len(STY['features'])}
hand-crafted features were competitive with them. Without the baselines we would have reported ~54% as a good number without
context. With ~10k training texts and embeddings learned from scratch, a CNN or LSTM has no advantage over sparse n-grams; we expect
pretrained encoders (e.g. a fine-tuned RoBERTa) to close the gap.</li>
<li><b>Debugging training matters more than architecture tweaks.</b> Our first CNN used dropout 0.5 directly on the embeddings and a
learning rate of 2e-3; it stopped after 3 epochs at 43% accuracy. Lowering the embedding dropout to 0.2 and the learning rate to 1e-3
gave +11 points. We learned to look at the learning curves (its validation F1 jumped between 0.31 and 0.39 from epoch to epoch, a clear warning sign)
before changing the model.</li>
<li><b>Compute is a design constraint.</b> All training ran on a 2-core CPU. The BiLSTM was about 4 to 6 times slower per epoch than
the CNN, so we halved its hidden size, capped its epochs (8 for RQ1, 6 elsewhere) and ran the RQ3/RQ4 sweeps with the CNN only; we ran 3 seeds
only for the main CNN result (RQ1). We built the experiment runner so each experiment writes its own JSON file and the queue can resume after an interruption,
which saved us when our machine restarted in the middle of a run.</li>
<li><b>Interpretability needs explicit experiments.</b> Accuracy alone did not tell us <i>why</i> the models work. The stylometric
analysis, the n-gram weights and especially the retraining ablations (removing formatting, keeping only function words, shuffling
words) were what turned the classifier into an explanation.</li>
<li><b>Limitations.</b> AlpacaEval outputs come from 2023 to 2024 model versions answering general instructions in English with
default settings, so our fingerprints may not transfer to newer versions, other languages, system prompts that request a style,
or edited text. Our coding subset is small (79 prompts), and most experiments use one seed.</li>
</ul>
"""


def _abstract():
    r = S["rq1"]; n = S["rq2"]["natural"]; c = S["rq2"]["confounded"]; _ab = S["rq4"]["ablation"]["CNN"]
    return (f"We study whether the authoring LLM can be identified from its text. We built a balanced dataset of "
            f"14,466 <code>(LLM_name, input, output)</code> triples from the public AlpacaEval outputs of 18 models in six families "
            f"(GPT, Claude, Llama, Qwen, Mistral, Google), all answering the same 805 prompts, and evaluated with prompt-grouped "
            f"splits. A CNN and a BiLSTM trained from scratch identify the family of unseen-prompt responses with "
            f"{pct(r['CNN']['acc_mean'])}% and {pct(r['BiLSTM']['acc_mean'])}% accuracy (chance 16.7%); a TF-IDF baseline reaches "
            f"{pct(r['TF-IDF+LR']['acc_mean'])}%. Claude, Llama and Google are easy to recognise; GPT and Qwen are often confused. "
            f"The prompt alone is useless on this data (chance accuracy) and adds little to the response, but in a simulated biased "
            f"collection the prompt alone reaches up to {pct(max(c['TF-IDF+LR']['input'], c['CNN']['input'] or 0, c['BiLSTM']['input'] or 0))}%, "
            f"exposing a common dataset confound. Fingerprints transfer to unseen prompt sources with a small loss and to coding prompts with a "
            f"somewhat larger loss (RQ3). Formatting (markdown bold, lists, headers), length and stock phrases are the strongest "
            f"interpretable signals. Retraining the CNN without formatting costs {100 * (_ab['original'] - _ab['noformat']):.1f} points, while "
            f"replacing every content word by a placeholder changes accuracy by only {100 * (_ab['funcwords'] - _ab['original']):+.1f}: "
            f"the fingerprint lies in how answers are structured, not in what they are about (RQ4).")


def SECTIONS():
    global ABSTRACT
    global RQ3_TEXT
    ABSTRACT = _abstract()
    RQ3_TEXT = _rq3_text()
    global ABLATION_TEXT
    ABLATION_TEXT = _abl_text()
    return [sec_front(), sec_data(), sec_models(), sec_rq1(), sec_rq2(), sec_rq3(), sec_rq4(), sec_lessons(), sec_refs()]



def _rq3_text():
    r = S["rq3"]; srcs = ["helpful_base", "koala", "oasst", "selfinstruct", "vicuna"]
    ind, cro = r["in_domain"], r["cross_domain"]
    def mean(d, keys):
        v = [d[k] for k in keys if d.get(k) is not None]
        return sum(v) / len(v) if v else None
    ci, cc = mean(ind["CNN"], srcs), mean(cro["CNN"], srcs)
    ti, tc = mean(ind["TF-IDF+LR"], srcs), mean(cro["TF-IDF+LR"], srcs)
    drops = {s: (cro["CNN"][s] - ind["CNN"][s]) for s in srcs if cro["CNN"].get(s) is not None}
    worst = min(drops, key=drops.get)
    pc = r["loso_per_class"]["CNN"]
    import numpy as _np
    fam_avg = _np.nanmean([[_np.nan if v is None else v for v in (pc[s] or [_np.nan] * 6)] for s in srcs], 0)
    order = _np.argsort(-fam_avg)
    fam_txt = ", ".join(f"{FAMILIES[i]} {fam_avg[i]:.2f}" for i in order)
    code_c = cro["CNN"].get("coding"); code_t = cro["TF-IDF+LR"].get("coding")
    cic = S.get("code_in_domain_cnn"); cit = r.get("code_in_domain_tfidf")
    return f"""
<p><b>i) Does performance decrease on unseen tasks?</b> Only slightly. Averaged over the five held-out sources, the CNN goes from
{pct(ci)}% in-domain to {pct(cc)}% cross-domain, and TF-IDF from {pct(ti)}% to {pct(tc)}% (Table 6, Figure 5). The largest drops are on <i>helpful_base</i> (CNN {100 * drops['helpful_base']:+.1f}, TF-IDF
{100 * (cro['TF-IDF+LR']['helpful_base'] - ind['TF-IDF+LR']['helpful_base']):+.1f} points) and <i>selfinstruct</i>
(CNN {100 * drops['selfinstruct']:+.1f}, TF-IDF {100 * (cro['TF-IDF+LR']['selfinstruct'] - ind['TF-IDF+LR']['selfinstruct']):+.1f}).
<i>helpful_base</i> is the easiest source in-domain, so part of its advantage was source-specific. <i>selfinstruct</i> is the most
distinctive source (short, templated tasks such as rewriting, classifying or extracting): its answers are short (median 140 words vs.
239 to 335 words for the other sources), so they carry fewer stylistic cues, and the other four sources contain fewer tasks of this kind.
Two sources even improve slightly for the CNN (koala and vicuna), which shows that most of the signal the model uses is not tied to one prompt
source. Note that the cross-domain models are trained on a similar number of prompts as the in-domain model (487 to 638 training prompts vs. 563),
so the comparison is largely not confounded by training-set size. For <b>coding</b> prompts, the CNN
trained without any coding prompt reaches {pct(code_c)}% (in-domain: {pct(cic)}%) and TF-IDF {pct(code_t)}% (in-domain: {pct(cit)}%).
Coding answers are harder for every model, even in-domain, likely because a large part of the answer is code, which looks similar across
LLMs. Removing all coding prompts from training changes accuracy on them by {100 * (code_c - cic):+.1f} points for the CNN and
{100 * (code_t - cit):+.1f} for TF-IDF (on a small test set of 306 responses). Even without any coding example the models stay far above
chance, which suggests that much of the fingerprint lives in the natural-language framing
around the code (openers, explanations, closing remarks).</p>
<p><b>ii) Are some models more consistently identifiable?</b> Yes. Averaged over the five LOSO folds, the per-family F1 of the CNN is:
{fam_txt}. The grouping is the same as in-domain (Table 4): the families with strong formatting habits transfer well, while GPT and Qwen
remain hard everywhere.</p>
<p><b>iii) General fingerprint or task-specific patterns?</b> Mostly a general fingerprint. If the classifier relied on task-specific
patterns, accuracy would collapse on an unseen source; instead it stays at roughly 3 times chance on every held-out source and on
coding. The remaining gap shows that part of the signal is task-dependent: how a model formats an answer depends on the task (a list for
"give me ideas", a code block for programming), so a model trained without a task type has not seen how each LLM handles it.</p>
"""


def _abl_text():
    a = S["rq4"]["ablation"]["CNN"]; L = S["rq4"]["length"]["CNN"]
    o = a["original"]
    d = lambda k: "n/a" if a.get(k) is None else f"{100 * (a[k] - o):+.1f}"
    Lk = {int(k): v for k, v in L.items() if v is not None}
    lens = sorted(Lk)
    curve = ", ".join(f"{k} tokens: {pct(Lk[k])}%" for k in lens)
    return f"""
<p>To test causally which signals matter, we transformed the text and <b>retrained</b> the CNN from scratch on the transformed data
(Table 8, Figure 6a). Retraining matters: simply feeding altered text to the original model would mainly measure distribution shift,
whereas retraining asks whether the information is still <i>learnable</i>.</p>
<ul>
<li><b>Removing formatting</b> (markdown symbols, list markers and all line breaks) costs {-float(d('noformat')):.1f} points, a large drop. Formatting is a major but not the only signal: the model still reaches {pct(a.get('noformat'))}%
(about {a.get('noformat', 0) / (1 / 6):.1f} times chance) on plain running text.</li>
<li><b>Keeping only function words and punctuation</b> (every content word replaced by the same placeholder <code>W</code>) changes accuracy by only {d('funcwords')} points, well within the CNN's seed-to-seed variation. What a response is <i>about</i> is almost irrelevant; how it is <i>put together</i>
(function words, punctuation, line structure, sentence rhythm) carries nearly all the information the CNN uses.</li>
<li><b>Keeping only content words</b> (lower-cased, no stop words, no punctuation, no line breaks) gives {pct(a.get('contentonly'))}%
({d('contentonly')}), the largest drop of all. This model learned slowly (training loss was still 1.71 when early stopping triggered), so the number is a lower bound, but vocabulary choice alone is clearly a much weaker signal than structure.</li>
<li><b>Removing the first line or sentence</b> changes accuracy by {d('noopener')} points. Stock openers ("Certainly!", "Here are",
"What a great question!") are very visible in Table 7, but the model can recover most of the signal from the rest of the response.</li>
<li><b>Shuffling the word order</b> costs {-float(d('shuffle')):.1f} points: local order (the n-grams detected by the convolution filters) matters, but a
bag of tokens still identifies the family {a.get('shuffle', 0) / (1 / 6):.1f} times better than chance.</li>
</ul>
<p><b>How early does the fingerprint appear?</b> Retraining the CNN on only the first k tokens of each response gives: {curve}
(Figure 6b). Even the first 16 tokens (roughly the opening sentence) identify the family at more than twice chance, most of the gain is reached by 64 tokens, and the curve is flat between 128 and 256 tokens. The fingerprint is therefore front-loaded (openers, the first heading or list), and longer CNN inputs are unlikely to help much.</p>
<p><b>Summary for RQ4.</b> i) The classifier relies on formatting habits (bold, lists, headers, line breaks), response length and
characteristic phrases, and on the arrangement of function words and punctuation rather than on topic words. ii) The signals are both
structural and linguistic: structure is the strongest single group, but removing all formatting still leaves a clear linguistic
fingerprint. iii) Removing formatting reduces accuracy substantially (by {-float(d('noformat')):.1f} points), while removing content words barely
matters.</p>
"""
