import argparse
import json
import pickle
import warnings

import torch
import yaml

from lrrnn import info
from lrrnn.models import POSTaggingRNN, WordSegmentationRNN
from lrrnn.taggers import LRRNNTagger, LRTagger, RNNTagger
from lrrnn.tokenizers import LRRNNTokenizer, LRTokenizer, RNNTokenizer


def load_pkl(file):
    with open(file, "rb") as f:
        return pickle.load(f)


def load_lr_tokenizer(vectorizer_file, estimator_file):
    vectorizer = load_pkl(vectorizer_file)
    estimator = load_pkl(estimator_file)
    return LRTokenizer(vectorizer, estimator)


def load_lr_tagger(vectorizers, estimators_file, ttoi_itot_file):
    estimators = load_pkl(estimators_file)
    with open(ttoi_itot_file, "r") as f:
        data = json.load(f)
        ttois = data["ttois"]
        itots = data["itots"]
    return LRTagger(estimators, vectorizers, ttois, itots)


def load_lr_taggers(vectorizers_file, estimators_files, ttoi_itot_files):
    vectorizers = load_pkl(vectorizers_file)
    taggers = []
    if len(estimators_files) != len(ttoi_itot_files):
        raise ValueError(
            "The number of estimators files and ttoi_itot files must be the same"
        )
    for estimators_file, ttoi_itot_file in zip(estimators_files, ttoi_itot_files):
        taggers.append(load_lr_tagger(vectorizers, estimators_file, ttoi_itot_file))
    return taggers


def load_rnn_tokenizer(checkpoint_file, model_config_file, device):
    with open(model_config_file, "r") as f:
        model_config = json.load(f)
    model = WordSegmentationRNN(
        len(model_config["char1_ttoi"]),
        len(model_config["char2_ttoi"]),
        len(model_config["char3_ttoi"]),
        model_config["char_embedding_dim"],
        len(model_config["chartype1_ttoi"]),
        len(model_config["chartype2_ttoi"]),
        len(model_config["chartype3_ttoi"]),
        model_config["chartype_embedding_dim"],
        model_config["hidden_size"],
        2,
    )
    model.load_state_dict(torch.load(checkpoint_file, weights_only=True))
    model.to(device)
    model.eval()

    return RNNTokenizer(
        model_config["char1_ttoi"],
        model_config["char2_ttoi"],
        model_config["char3_ttoi"],
        model_config["chartype1_ttoi"],
        model_config["chartype2_ttoi"],
        model_config["chartype3_ttoi"],
        model,
        device,
    )


def load_rnn_tagger(checkpoint_file, model_config_file, device):
    with open(model_config_file, "r") as f:
        model_config = json.load(f)
    model = POSTaggingRNN(
        len(model_config["char1_ttoi"]),
        len(model_config["char2_ttoi"]),
        len(model_config["char3_ttoi"]),
        model_config["char_embedding_dim"],
        len(model_config["chartype1_ttoi"]),
        len(model_config["chartype2_ttoi"]),
        len(model_config["chartype3_ttoi"]),
        model_config["chartype_embedding_dim"],
        len(model_config["word_ttoi"]),
        model_config["word_embedding_dim"],
        model_config["hidden_size"],
        len(model_config["tag_to_id"]),
    )
    model.load_state_dict(torch.load(checkpoint_file, weights_only=True))
    model.to(device)
    model.eval()

    return RNNTagger(
        model_config["char1_ttoi"],
        model_config["char2_ttoi"],
        model_config["char3_ttoi"],
        model_config["chartype1_ttoi"],
        model_config["chartype2_ttoi"],
        model_config["chartype3_ttoi"],
        model_config["word_ttoi"],
        model,
        device,
        model_config["tag_to_id"],
        model_config["id_to_tag"],
    )


def load_rnn_taggers(checkpoint_files, model_config_files, device):
    taggers = []
    if len(checkpoint_files) != len(model_config_files):
        raise ValueError(
            "The number of checkpoint files and model_config files must be the same"
        )
    for checkpoint_file, model_config_file in zip(checkpoint_files, model_config_files):
        taggers.append(load_rnn_tagger(checkpoint_file, model_config_file, device))
    return taggers


def validate_config(config):
    if "lr_tokenizer" not in config:
        raise ValueError("The configuration must contain the 'lr_tokenizer' key")
    lr_tokenizer_config = config["lr_tokenizer"]
    if "vectorizer" not in lr_tokenizer_config:
        raise ValueError("The configuration must contain the 'vectorizer' key")
    if "estimator" not in lr_tokenizer_config:
        raise ValueError("The configuration must contain the 'estimator' key")

    if "lr_taggers" not in config:
        raise ValueError("The configuration must contain the 'lr_taggers' key")
    lr_taggers_config = config["lr_taggers"]
    if "vectorizer" not in lr_taggers_config:
        raise ValueError("The configuration must contain the 'vectorizer' key")
    if "estimators" not in lr_taggers_config:
        raise ValueError("The configuration must contain the 'estimators' key")
    if "mappings" not in lr_taggers_config:
        raise ValueError("The configuration must contain the 'mappings' key")
    num_lr_taggers = len(lr_taggers_config["estimators"])
    if num_lr_taggers != len(lr_taggers_config["mappings"]):
        raise ValueError(
            "The number of estimators and mappings must be the same in the 'lr_taggers' key"
        )

    if "rnn_tokenizer" not in config:
        raise ValueError("The configuration must contain the 'rnn_tokenizer' key")
    rnn_tokenizer_config = config["rnn_tokenizer"]
    if "checkpoint" not in rnn_tokenizer_config:
        raise ValueError("The configuration must contain the 'checkpoint' key")
    if "model_config" not in rnn_tokenizer_config:
        raise ValueError("The configuration must contain the 'model_config' key")

    if "rnn_taggers" not in config:
        raise ValueError("The configuration must contain the 'rnn_taggers' key")
    rnn_taggers_config = config["rnn_taggers"]
    if "checkpoints" not in rnn_taggers_config:
        raise ValueError("The configuration must contain the 'checkpoints' key")
    if "model_configs" not in rnn_taggers_config:
        raise ValueError("The configuration must contain the 'model_configs' key")
    num_rnn_taggers = len(rnn_taggers_config["checkpoints"])
    if num_rnn_taggers != len(rnn_taggers_config["model_configs"]):
        raise ValueError(
            "The number of checkpoints and model_configs must be the same in the 'rnn_taggers' key"
        )

    if num_lr_taggers != num_rnn_taggers:
        raise ValueError(
            "The number of LR taggers and RNN taggers must be the same in the configuration"
        )

    if "alpha" not in config:
        raise ValueError("The configuration must contain the 'alpha' key")
    if not isinstance(config["alpha"], float):
        raise ValueError("The 'alpha' key must be a float")
    if config["alpha"] < 0 or 1 < config["alpha"]:
        raise ValueError("The 'alpha' key must be in the range [0, 1]")

    if "beta" not in config:
        raise ValueError("The configuration must contain the 'beta' key")
    if not isinstance(config["beta"], float):
        raise ValueError("The 'beta' key must be a float")
    if config["beta"] < 0 or 1 < config["beta"]:
        raise ValueError("The 'beta' key must be in the range [0, 1]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="An implementation of the [LR+RNN] analyzer, which performs word segmentation and part-of-speech tagging"
    )
    parser.add_argument("config", help="path to yaml file containing configuration")
    parser.add_argument("text", help="path to text file to analyze")
    parser.add_argument(
        "--use_cuda",
        action="store_true",
        help="use CUDA if available",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    validate_config(config)

    if args.use_cuda:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            warnings.warn("CUDA is not available. Using CPU instead.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    lr_tokenizer = load_lr_tokenizer(
        config["lr_tokenizer"]["vectorizer"], config["lr_tokenizer"]["estimator"]
    )
    lr_taggers = load_lr_taggers(
        config["lr_taggers"]["vectorizer"],
        config["lr_taggers"]["estimators"],
        config["lr_taggers"]["mappings"],
    )
    rnn_tokenizer = load_rnn_tokenizer(
        config["rnn_tokenizer"]["checkpoint"],
        config["rnn_tokenizer"]["model_config"],
        device,
    )
    rnn_taggers = load_rnn_taggers(
        config["rnn_taggers"]["checkpoints"],
        config["rnn_taggers"]["model_configs"],
        device,
    )

    alpha = config["alpha"]
    beta = config["beta"]
    tokenizer = LRRNNTokenizer(lr_tokenizer, rnn_tokenizer, alpha)
    taggers = []
    for i in range(len(lr_taggers)):
        taggers.append(LRRNNTagger(lr_taggers[i], rnn_taggers[i], beta))

    with open(args.text) as f:
        for line in f:
            line = line.replace(info.newline, "")

            # Ignore the empty line
            if line == "":
                continue

            tokens = tokenizer.predict(line)
            tagger_predictions = [tagger.predict(tokens) for tagger in taggers]

            for i, morph in enumerate(zip(tokens, *tagger_predictions)):
                print(info.sep.join(morph), end="")
                if i == len(tokens) - 1:
                    print(end=info.newline)
                else:
                    print(end=" ")
