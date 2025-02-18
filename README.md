# LR-RNN

LR-RNN is a Python implementation of a morphological analysis algorithm introduced in the paper *"Sequential Morphological Analysis of Hiragana Strings using Recurrent Neural Network and Logistic Regression"*.

This implementation enables you to reproduce the experimental results presented in the paper.

The proposed method specializes in analyzing highly ambiguous sequences, particularly Japanese text written in hiragana. Compared to conventional analyzers, LR-RNN achieves superior performance in both word segmentation and part-of-speech tagging when processing hiragana sentences. Below are the key evaluation results from the paper:

**Word Segmentation:**
| Analyzer |   F-measure (Precision/Recall)  |
|  :---    |             ---:                |
|  MeCab   |       91.92 (90.84/93.03)       |
|  KyTea   |       91.93 (92.11/91.74)       |
|  LR-RNN  | **94.37** (**94.35**/**94.39**) |

**Word Segmentation & Part-of-Speech Tagging:**
| Analyzer |   F-measure (Precision/Recall)  |
|  :---    |             ---:                |
|  MeCab   |       77.90 (76.98/78.84)       |
|  KyTea   |       86.13 (86.30/85.95)       |
|  LR-RNN  | **88.22** (**88.20**/**88.23**) |

### Citation

> [*Sequential Morphological Analysis of Hiragana Strings using Recurrent Neural Network and Logistic Regression*](https://doi.org/10.5715/jnlp.29.367)\
> Authors: [Shuhei Moriyama](https://researchmap.jp/shuhei-moriyama), [Tomohiro Ohno](https://researchmap.jp/7000019470)\
> Published in: *Journal of Natural Language Processing*, 2022, Volume 29, Issue 2, Pages 367-394.\
> *Language: Japanese*

## Installation

### Preparing the environment

To set up LR-RNN, you need:

- Docker
- NVIDIA Container Toolkit

### Building a Docker image

1. Navigate to the `docker/` directory in the repository:

```bash
cd docker/
```

2. Build a Docker image:

```bash
docker build -t lrrnn .
```

This command creates a Docker image named `lrrnn`.

### Starting a Docker container

1. Verify your current working directory is `LR-RNN/`:

```bash
pwd
```

2. Run the following command to start a Docker container:

```bash
docker run --gpus=all --rm -it -v "$(pwd)":/workspace -p 6006:6006 lrrnn
```

> [!Note]
> Port 6006 is exposed for TensorBoard visualization.

3. Check if the files and directories in the current working directory are mounted to the `/workspace` directory within the container:

```bash
ls /workspace
```

All commands in the following sections should be executed within this container.

## Reproduce the results in the paper

The steps below outline the process for replicating the results presented in Section 3 of the paper.

### Preparing the Kyoto University Text Corpus

[Kyoto University Text Corpus Version 4.0](https://nlp.ist.i.kyoto-u.ac.jp/EN/?Kyoto%20University%20Text%20Corpus) is required to construct the dataset used in the experiments in the paper. For more information on the Kyoto University Text Corpus, please refer to the link above.

Save the Kyoto University Text Corpus under the `LR-RNN/` directory:

```bash
LR-RNN
├── KyotoCorpus4.0
│   └── dat
│       └── syn
│           ├── 950101.KNP
│           ├── 950102.KNP
│           └── ...
├── README.md (this document)
└── ...
```

### Building the dataset

Long story short, run the following script to create the dataset:

```bash
./reproduction/scripts/build_datasets.sh
```

The experiment employs 5-fold cross-validation. Therefore, the script creates five datasets from the Kyoto University Text Corpus. Each dataset is created by extracting a different set of documents from the corpus.

After running the script, the following directory structure is created:

```bash
LR-RNN
├── data
│   ├── test_950112
│   ├── test_950113
│   ├── test_950114
│   ├── test_950115
│   └── test_950116
├── README.md (this document)
└── ...
```

Each dataset contains the following files:

```bash
test_950112
├── kana
│   ├── gt
│   │   ├── test.txt
│   │   ├── train.txt
│   │   └── valid.txt
│   └── raw
│       ├── test.txt
│       ├── train.txt
│       └── valid.txt
└── kanji
    ├── gt
    │   ├── test.txt
    │   ├── train.txt
    │   └── valid.txt
    └── raw
        ├── test.txt
        ├── train.txt
        └── valid.txt
```

- The `kana` and `kanji` directories contain the data in hiragana and kanji, respectively.
  - We use the `kana` data in the following sections.
- The `raw` directory contains the unsegmented text data, and the `gt` directory contains the segmented text data.
  - The `gt` directory is used for training and evaluation as the ground truth.
  - The `raw` directory is used for prediction as the input data.
- The `train.txt`, `valid.txt`, and `test.txt` files contain the training, validation, and test data, respectively.

### Dataset format

The ground truth files (`gt` directory) use the following format:
- One sentence per line (ends with `\n`)
- Morphemes separated by spaces
- Each morpheme contains seven fields separated by slashes:
  1. Surface form in hiragana
  2. Part-of-speech (Major category)
  3. Part-of-speech (Subcategory)
  4. Group
  5. Conjugation type
  6. Kanji representation
  7. Reading in hiragana

The files in the `raw` directory are formatted as follows:

```
たろうはとうきょうだいがくにいった
じろうはきょうとだいがくにいった
```

- Each line ends with the newline character `\n`.
- Each line contains an unsegmented sequence of characters.

### Training

> [!NOTE]
> Training takes approximately 8 hours using an Intel Core i7-12700K CPU and RTX 3080 Ti GPU. You can alternatively download pre-trained models from [Releases](https://github.com/kanpachi1/LR-RNN/releases).

To train the proposed method using the datasets created in the previous section, run `reproduction/scripts/train.sh`.

```bash
./reproduction/scripts/train.sh
```

The script runs 5 sets of training on each dataset.
When the script finishes, trained model files are output to the `data/<each-dataset-name>/models/` directory.

The script also outputs training logs for neural networks to the `models/` directory. The log files are named `events.out.tfevents.*`. You can visualize the training progress using TensorBoard by running:

```bash
tensorboard --host=0.0.0.0 --logdir=data/test_950112/models/<directory-starting-with-rnn>
```

Then access TensorBoard through your browser at http://localhost:6006.

### Prediction

To analyze each test data using the trained models, run `reproduction/scripts/predict.sh`.

```bash
./reproduction/scripts/predict.sh
```

After the script finishes, prediction results are output to the `predictions/` directory.

```bash
LR-RNN
├── predictions
│   ├── test_950112_kana.txt
│   ├── test_950113_kana.txt
│   ├── test_950114_kana.txt
│   ├── test_950115_kana.txt
│   └── test_950116_kana.txt
└── ...
```

### Evaluation

`reproduction/scripts/evaluate.sh` evaluates the prediction results.

```bash
./reproduction/scripts/evaluate.sh
```

The script outputs Precision, Recall, and F-measure on each dataset to the standard output.
The average of the evaluation results will be similar to the results in the paper.

That's it! You have successfully reproduced the results in the paper.

## License

- The contents of this repository, excluding the pre-trained model files, are licensed under the MIT License.
- The pre-trained model files in [Releases](https://github.com/kanpachi1/LR-RNN/releases) are available for research purposes only.
- See `LICENSE` for the complete MIT License terms.
