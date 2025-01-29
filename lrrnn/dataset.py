from enum import Enum

from . import info


def parse_syn(knp_file):
    """
    Parses KNP file and extracts sentences with their annotations.

    Args:
        knp_file (str): Path to KNP file in KyotoCorpus4.0/dat/syn/ directory.
    Returns:
        list of list[dict]: List of sentences.
            Each sentence is represented as a list of morphemes.
            Each morpheme contains the following keys:
            - 'surface': The surface form of the word.
            - 'yomi': The reading of the word.
            - 'base_form': The base form of the word.
            - 'pos1': The primary part of speech.
            - 'pos2': The secondary part of speech.
            - 'group': The group of the word.
            - 'conjugation': The conjugation form of the word.
    """
    sentences = []
    with open(knp_file, encoding="euc-jp") as f:
        sentence = []
        for line in f:
            line = line.replace("\n", "")
            if line.startswith("# "):
                continue
            elif line.startswith("* "):
                continue
            elif line == "EOS":
                sentences.append(sentence)
                sentence = []
            else:
                features = line.split()
                sentence.append(
                    dict(
                        surface=features[0],
                        yomi=features[1],
                        base_form=features[2],
                        pos1=features[3],
                        pos2=features[4],
                        group=features[5],
                        conjugation=features[6],
                    )
                )
    return sentences


def load_corpus(knp_file_names, corpus_dir):
    """Loads KNP files under syn/ directory in the Kyoto University Text Corpus Version 4.0.

    Args:
        knp_file_names (list[str]): List of KNP file names to be loaded.
        corpus_dir (pathlib.Path): Path to Kyoto University Text Corpus Version 4.0 directory.
    Returns:
        list of list[dict]: List of sentences.
            Each sentence is represented as a list of morphemes.
    """
    sentences = []
    for name in knp_file_names:
        sentences.extend(parse_syn(corpus_dir / "dat" / "syn" / name))
    return sentences


class CorpusType(Enum):
    KANA = 1
    KANJI = 2
    UNSEGMENTED_KANA = 3
    UNSEGMENTED_KANJI = 4


def format_morpheme(morph, corpus_type):
    """Formats morpheme into a string.

    Args:
        morph (dict): Morpheme information.
        corpus_type (CorpusType): Corpus type.
    Returns:
        str: Formatted morpheme.
    """
    if corpus_type == CorpusType.KANA:
        return info.sep.join(
            (
                morph["yomi"],
                morph["pos1"],
                morph["pos2"],
                morph["group"],
                morph["conjugation"],
                morph["surface"],
                morph["yomi"],
            )
        )
    elif corpus_type == CorpusType.KANJI:
        return info.sep.join(
            (
                morph["surface"],
                morph["pos1"],
                morph["pos2"],
                morph["group"],
                morph["conjugation"],
                morph["surface"],
                morph["yomi"],
            )
        )
    elif corpus_type == CorpusType.UNSEGMENTED_KANA:
        return morph["yomi"]
    elif corpus_type == CorpusType.UNSEGMENTED_KANJI:
        return morph["surface"]
    else:
        raise ValueError(f"Unsupported CorpusType: {corpus_type}")


def format_sentence(sentence, corpus_type):
    """Formats sentence into a string.

    Args:
        sentence (list[dict]): List of morphemes.
        corpus_type (CorpusType): Corpus type.
    Returns:
        str: Formatted sentence.
    """
    if corpus_type == CorpusType.KANA:
        return " ".join([format_morpheme(morph, corpus_type) for morph in sentence])
    elif corpus_type == CorpusType.KANJI:
        return " ".join([format_morpheme(morph, corpus_type) for morph in sentence])
    elif corpus_type == CorpusType.UNSEGMENTED_KANA:
        return "".join([format_morpheme(morph, corpus_type) for morph in sentence])
    elif corpus_type == CorpusType.UNSEGMENTED_KANJI:
        return "".join([format_morpheme(morph, corpus_type) for morph in sentence])
    else:
        raise ValueError(f"Unsupported CorpusType: {corpus_type}")


def save_sentences_to_file(sentences, file_path, corpus_type):
    """Saves sentences to a file.

    Args:
        sentences (list[list[dict]]): List of sentences.
        file_path (str): Path to the output file.
        corpus_type (CorpusType): Corpus type.
    """
    with open(file_path, "w") as f:
        for morphs in sentences:
            f.write(format_sentence(morphs, corpus_type))
            f.write(info.newline)
