#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

OUTPUT_DIR="predictions"

for DATASET in "test_950112" "test_950113" "test_950114" "test_950115" "test_950116"
do
    echo "Evaluating on ${DATASET}"
    python lrrnn/bin/evaluate.py ${OUTPUT_DIR}/${DATASET}_kana.txt data/${DATASET}/kana/gt/test.txt
done
