#!/bin/bash
# Runs every neural experiment reported in the paper, in the order we ran them.
# Results are cached in results/<name>.json, so the script can be re-run to resume after an interruption.
# (On a 2-core CPU this takes about 4 hours; BiLSTM runs dominate.)
set -e
EPOCHS_CNN=12 EPOCHS_LSTM=8 python3 src/run_experiments.py rq1_cnn_s0 rq1_lstm_s0
export EPOCHS_CNN=12 EPOCHS_LSTM=6
for e in rq2_cnn_input rq2_cnn_both rq2_lstm_input \
         rq2conf_cnn_input rq2conf_cnn_output rq2conf_cnn_both \
         rq3_code_cnn rq2_lstm_both rq2conf_lstm_input rq2conf_lstm_output rq2conf_lstm_both \
         rq3_loso_cnn_helpful_base rq3_loso_cnn_koala rq3_loso_cnn_oasst rq3_loso_cnn_selfinstruct rq3_loso_cnn_vicuna \
         rq1_cnn_s1 rq1_cnn_s2 \
         rq4_noformat_cnn rq4_funcwords_cnn rq4_shuffle_cnn rq4_noopener_cnn rq4_contentonly_cnn \
         rq4_len16_cnn rq4_len32_cnn rq4_len64_cnn rq4_len128_cnn; do
  python3 src/run_experiments.py $e
done
# Optional extras (defined in src/run_experiments.py, not all reported):
# rq1_model18_cnn rq3_code_lstm rq4_noformat_lstm rq4_shuffle_lstm rq4_funcwords_lstm rq1_lstm_s1 rq1_lstm_s2 rq3_loso_lstm_*
