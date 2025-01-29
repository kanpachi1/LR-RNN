#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

for DATASET in "test_950112" "test_950113" "test_950114" "test_950115" "test_950116"
do
    echo "Training on ${DATASET}"
    python lrrnn/bin/train_lr_tokenizer.py reproduction/configs/${DATASET}/train_lr_tokenizer.yaml
    python lrrnn/bin/train_lr_taggers.py reproduction/configs/${DATASET}/train_lr_taggers.yaml
    python lrrnn/bin/train_rnn_tokenizer.py reproduction/configs/${DATASET}/train_rnn_tokenizer.yaml
    python lrrnn/bin/train_rnn_tagger.py reproduction/configs/${DATASET}/train_rnn_tagger_1.yaml
    python lrrnn/bin/train_rnn_tagger.py reproduction/configs/${DATASET}/train_rnn_tagger_2.yaml
    python lrrnn/bin/train_rnn_tagger.py reproduction/configs/${DATASET}/train_rnn_tagger_3.yaml
    python lrrnn/bin/train_rnn_tagger.py reproduction/configs/${DATASET}/train_rnn_tagger_4.yaml
    python lrrnn/bin/train_rnn_tagger.py reproduction/configs/${DATASET}/train_rnn_tagger_5.yaml
    python lrrnn/bin/train_rnn_tagger.py reproduction/configs/${DATASET}/train_rnn_tagger_6.yaml
done
