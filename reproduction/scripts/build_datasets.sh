#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

KYOTO_CORPUS_PATH="KyotoCorpus4.0"

for TEST_DATA in "950112" "950113" "950114" "950115" "950116"
do
    echo "Building dataset with ${TEST_DATA}.KNP as test data"
    python lrrnn/bin/build_dataset.py ${KYOTO_CORPUS_PATH} ${TEST_DATA}.KNP --output data/test_${TEST_DATA}
done
