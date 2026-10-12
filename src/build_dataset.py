"""Build the (LLM_name, LLM_input, LLM_output) dataset from AlpacaEval public model outputs.

Source: https://github.com/tatsu-lab/alpaca_eval  (results/<model>/model_outputs.json)
Download raw files with scripts/download_data.sh, then run this script.
"""
import json, os, re, sys
import pandas as pd

FAMILIES = {
    "GPT":     ["gpt-3.5-turbo-0613", "gpt4_1106_preview", "gpt-4o-2024-05-13"],
    "Claude":  ["claude-2.1", "claude-3-opus-20240229", "claude-3-5-sonnet-20240620"],
    "Llama":   ["llama-2-70b-chat-hf", "Meta-Llama-3-70B-Instruct", "Meta-Llama-3.1-405B-Instruct-Turbo"],
    "Qwen":    ["Qwen1.5-72B-Chat", "Qwen1.5-110B-Chat", "Qwen2-72B-Instruct"],
    "Mistral": ["Mistral-7B-Instruct-v0.2", "Mixtral-8x7B-Instruct-v0.1", "mistral-large-2402"],
    "Google":  ["gemini-pro", "gemma-7b-it", "gemma-2b-it"],
}

def main(raw_dir="data/raw", out="data/llm_dataset.csv"):
    rows = []
    for fam, models in FAMILIES.items():
        for m in models:
            for r in json.load(open(os.path.join(raw_dir, f"{m}.json"))):
                out_txt = (r["output"] or "").strip()
                if len(out_txt) == 0:
                    continue
                rows.append(dict(family=fam, model=m, source=r["dataset"],
                                 input=r["instruction"].strip(), output=out_txt))
    df = pd.DataFrame(rows)
    # prompt id shared across models -> used for group-wise splitting (no prompt leakage)
    df["prompt_id"] = df["input"].astype("category").cat.codes
    df.to_csv(out, index=False)
    print(df.groupby(["family"]).size())
    print(df.groupby(["source"]).prompt_id.nunique())
    print("rows", len(df), "unique prompts", df.prompt_id.nunique())

if __name__ == "__main__":
    main(*sys.argv[1:])
