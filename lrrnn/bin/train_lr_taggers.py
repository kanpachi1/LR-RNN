import argparse
import json
import os
import pickle

from scipy.sparse import vstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import yaml

from lrrnn import data_loader
from lrrnn.taggers import LRTagger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to yaml file containing configuration")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("Loading training data...")
    sentences, Tags = data_loader.load_corpus_data_for_pos_tagging(config["train"])

    for level in range(1, len(Tags) + 1):
        os.makedirs(os.path.join(config["savedir"], "tagger_{}".format(level)))

    print("Extracting features...")
    vocab = [*{*[t for tokens in sentences for t in tokens]}]
    vectorizers = {v: DictVectorizer() for v in vocab}
    features = {v: [] for v in vocab}
    for tokens in sentences:
        for i, t in enumerate(tokens):
            features[t].append(LRTagger.extract_features(tokens, i))

    print("Vectorizing features...")
    for v, vectorizer in vectorizers.items():
        vectorizer.fit(features[v])
    Xs = {v: None for v in vocab}
    for v, feat in features.items():
        vectorizer = vectorizers[v]
        if Xs[v] is None:
            Xs[v] = vectorizer.transform(feat)
        else:
            Xs[v] = vstack((Xs[v], vectorizer.transform(feat)))
    del features

    print("Number of columns of morpheme information:", len(Tags))
    for level, tags in enumerate(Tags, start=1):
        print("Training model at column {}...".format(level))

        # Build Ys (groundtruth) for each vocab
        Ys = {v: [] for v in vocab}
        for i, t in enumerate([t for tokens in sentences for t in tokens]):
            Ys[t].append(tags[i])

        ttois = {v: {} for v in vocab}
        itots = {v: [] for v in vocab}
        estimators = {v: None for v in vocab}
        for v in vocab:
            # Build token-to-index and index-to-token mappings for each vocab
            for i, tag in enumerate([*{*Ys[v]}]):
                ttois[v][tag] = i
                itots[v].append(tag)
            # Train the model for each vocab
            if len(ttois[v]) > 1:
                estimators[v] = LogisticRegression(
                    solver=config["solver"], max_iter=config["max_iter"]
                )
                y = [ttois[v][tag] for tag in Ys[v]]
                estimators[v].fit(Xs[v], y)

        savedir = os.path.join(config["savedir"], "tagger_{}".format(level))
        with open(os.path.join(savedir, "ttoi_itot.json"), "w") as f:
            json.dump({"ttois": ttois, "itots": itots}, f)
        with open(os.path.join(savedir, "estimators.pkl"), "wb") as f:
            pickle.dump(estimators, f)
        print("Saved model files to", savedir)

    with open(os.path.join(config["savedir"], "vectorizers.pkl"), "wb") as f:
        pickle.dump(vectorizers, f)
    print("Saved vectorizers to", config["savedir"])
