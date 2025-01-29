import argparse
from pathlib import Path

from lrrnn.dataset import (
    CorpusType,
    load_corpus,
    save_sentences_to_file,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script divides the Kyoto University Text Corpus Version 4.0 into train, validation, and test data to create an experimental dataset."
    )
    parser.add_argument(
        "corpus", help="path to Kyoto University Text Corpus Version 4.0 directory"
    )
    parser.add_argument(
        "test",
        help="comma-separated KNP file names to be used as test data (e.g., 950112.KNP,950116.KNP)",
    )
    parser.add_argument(
        "--valid",
        default="950117.KNP",
        help="comma-separated KNP file names to be used as validation data (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="data",
        help="Output directory for experimental dataset (default: %(default)s)",
    )
    args = parser.parse_args()

    # All KNP files in the KyotoCorpus4.0/dat/syn/ directory
    KNP_FILES = {
        "950101.KNP",
        "950103.KNP",
        "950104.KNP",
        "950105.KNP",
        "950106.KNP",
        "950107.KNP",
        "950108.KNP",
        "950109.KNP",
        "950110.KNP",
        "950111.KNP",
        "950112.KNP",
        "950113.KNP",
        "950114.KNP",
        "950115.KNP",
        "950116.KNP",
        "950117.KNP",
        "9501ED.KNP",
        "9502ED.KNP",
        "9503ED.KNP",
        "9504ED.KNP",
        "9505ED.KNP",
        "9506ED.KNP",
        "9507ED.KNP",
        "9508ED.KNP",
        "9509ED.KNP",
        "9510ED.KNP",
        "9511ED.KNP",
        "9512ED.KNP",
    }
    test = sorted(args.test.split(","))
    valid = sorted(args.valid.split(","))
    train = sorted(KNP_FILES - (set(test) | set(valid)))

    corpus_dir = Path(args.corpus)

    train_sentences = load_corpus(train, corpus_dir)
    valid_sentences = load_corpus(valid, corpus_dir)
    test_sentences = load_corpus(test, corpus_dir)

    groundtruth_kana = Path(args.output) / "kana" / "gt"
    groundtruth_kana.mkdir(parents=True)
    groundtruth_kanji = Path(args.output) / "kanji" / "gt"
    groundtruth_kanji.mkdir(parents=True)
    unsegmented_kana = Path(args.output) / "kana" / "raw"
    unsegmented_kana.mkdir(parents=True)
    unsegmented_kanji = Path(args.output) / "kanji" / "raw"
    unsegmented_kanji.mkdir(parents=True)

    save_sentences_to_file(
        train_sentences, groundtruth_kana / "train.txt", CorpusType.KANA
    )
    save_sentences_to_file(
        valid_sentences, groundtruth_kana / "valid.txt", CorpusType.KANA
    )
    save_sentences_to_file(
        test_sentences, groundtruth_kana / "test.txt", CorpusType.KANA
    )
    save_sentences_to_file(
        train_sentences, groundtruth_kanji / "train.txt", CorpusType.KANJI
    )
    save_sentences_to_file(
        valid_sentences, groundtruth_kanji / "valid.txt", CorpusType.KANJI
    )
    save_sentences_to_file(
        test_sentences, groundtruth_kanji / "test.txt", CorpusType.KANJI
    )
    save_sentences_to_file(
        train_sentences, unsegmented_kana / "train.txt", CorpusType.UNSEGMENTED_KANA
    )
    save_sentences_to_file(
        valid_sentences, unsegmented_kana / "valid.txt", CorpusType.UNSEGMENTED_KANA
    )
    save_sentences_to_file(
        test_sentences, unsegmented_kana / "test.txt", CorpusType.UNSEGMENTED_KANA
    )
    save_sentences_to_file(
        train_sentences, unsegmented_kanji / "train.txt", CorpusType.UNSEGMENTED_KANJI
    )
    save_sentences_to_file(
        valid_sentences, unsegmented_kanji / "valid.txt", CorpusType.UNSEGMENTED_KANJI
    )
    save_sentences_to_file(
        test_sentences, unsegmented_kanji / "test.txt", CorpusType.UNSEGMENTED_KANJI
    )
