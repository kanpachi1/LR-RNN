import argparse
import os
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import yaml

from lrrnn import data_loader
from lrrnn.feature_extraction import (
    extract_char_from_surface,
    extract_chartype_from_surface,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to yaml file containing configuration")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs(config["savedir"])

    print("Loading training data...")
    unsegmented_sentences, labels = data_loader.load_corpus_data_for_word_segmentation(
        config["train"]
    )

    print("Extracting features...")
    feature_dicts = []
    for surface in unsegmented_sentences:
        for c1, c2, c3, ct1, ct2, ct3 in zip(
            extract_char_from_surface(surface, 1, 3),
            extract_char_from_surface(surface, 2, 3),
            extract_char_from_surface(surface, 3, 3),
            extract_chartype_from_surface(surface, 1, 3),
            extract_chartype_from_surface(surface, 2, 3),
            extract_chartype_from_surface(surface, 3, 3),
        ):
            feature_dicts.append(dict(**c1, **c2, **c3, **ct1, **ct2, **ct3))
    del unsegmented_sentences

    print("Vectorizing features...")
    vectorizer = DictVectorizer()
    vectorizer.fit(feature_dicts)
    X = vectorizer.transform(feature_dicts)
    del feature_dicts

    print("Training model...")
    lr = LogisticRegression(solver=config["solver"], max_iter=config["max_iter"])
    lr.fit(X, labels)

    with open(os.path.join(config["savedir"], "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(config["savedir"], "estimator.pkl"), "wb") as f:
        pickle.dump(lr, f)
    print("Saved model files to", config["savedir"])
