# Who Wrote It? Identifying LLMs from Their Responses

CMPSC 448 midterm project (individual). Team leader: **Younsoo Park**.

We classify which LLM family (GPT, Claude, Llama, Qwen, Mistral, Google Gemini/Gemma) produced a response,
using a **CNN** and a **bidirectional LSTM** trained from scratch in PyTorch, and study RQ1 to RQ4.
The full write-up is in [`report/Project_Report.pdf`](report/Project_Report.pdf).

## Data

`data/llm_dataset.csv` has 14,466 rows with the columns
`family, model, source, input, output, prompt_id` (that is, `(LLM_name, LLM_input, LLM_output)` plus metadata).

* Source: the public model outputs released with **AlpacaEval** (Li et al., 2023),
  `https://github.com/tatsu-lab/alpaca_eval`, files `results/<model>/model_outputs.json`
  (commit `cd543a1`). Every model answered the **same 805 instructions**, which come from five public
  instruction sets (`helpful_base`, `koala`, `oasst`, `selfinstruct`, `vicuna`).
* 6 families x 3 models = 18 models (see `src/build_dataset.py`). Empty responses are dropped.
* No private or personal data is involved: the prompts are public benchmark instructions.

Rebuild from scratch:

```bash
bash scripts/download_data.sh        # downloads the 18 raw json files into data/raw/
python src/build_dataset.py          # writes data/llm_dataset.csv
```

## Code

| File | Purpose |
|---|---|
| `src/common.py` | tokenizer (keeps newlines / markdown / case), vocabulary, prompt-grouped split, **TextCNN** and **BiLSTM** models, training loop with early stopping |
| `src/run_experiments.py` | all neural experiments (RQ1 to RQ4); each writes `results/<name>.json` |
| `src/baselines_and_analysis.py` | TF-IDF + logistic regression baselines, stylometric features, top n-grams (RQ4) |
| `src/make_figures.py` | figures in `figures/` and `results/summary.json` |
| `report/build_report.py` | builds `report/Project_Report.pdf` from the results |
| `run_queue.sh` | runs the full experiment queue in order |

## Reproduce

```bash
pip install torch pandas scikit-learn matplotlib playwright
bash run_queue.sh                      # all neural experiments (CPU is fine; ~4 h on 2 cores)
python src/baselines_and_analysis.py   # baselines + RQ4 analysis
python src/make_figures.py             # figures + summary.json
python report/build_report.py          # PDF report
python src/run_experiments.py --list   # list every experiment name
```

All splits are made **by prompt** (the same instruction never appears in two splits), with a fixed seed.
