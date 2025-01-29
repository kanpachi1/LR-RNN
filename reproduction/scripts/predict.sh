#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

OUTPUT_DIR="predictions"
mkdir ${OUTPUT_DIR}

for DATASET in "test_950112" "test_950113" "test_950114" "test_950115" "test_950116"
do
    echo "Predicting on ${DATASET}"
    python lrrnn/bin/analyze.py --use_cuda reproduction/configs/${DATASET}/analyzer.yaml data/${DATASET}/kana/raw/test.txt > ${OUTPUT_DIR}/${DATASET}_kana.txt
done
