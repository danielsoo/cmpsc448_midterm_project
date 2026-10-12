#!/bin/bash
# Download the public AlpacaEval model outputs used in this project.
set -e
mkdir -p data/raw
BASE=https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/main/results
for m in gpt-3.5-turbo-0613 gpt4_1106_preview gpt-4o-2024-05-13 \
         claude-2.1 claude-3-opus-20240229 claude-3-5-sonnet-20240620 \
         llama-2-70b-chat-hf Meta-Llama-3-70B-Instruct Meta-Llama-3.1-405B-Instruct-Turbo \
         Qwen1.5-72B-Chat Qwen1.5-110B-Chat Qwen2-72B-Instruct \
         Mistral-7B-Instruct-v0.2 Mixtral-8x7B-Instruct-v0.1 mistral-large-2402 \
         gemini-pro gemma-7b-it gemma-2b-it; do
  curl -sSfL -o "data/raw/$m.json" "$BASE/$m/model_outputs.json"
  echo "downloaded $m"
done
