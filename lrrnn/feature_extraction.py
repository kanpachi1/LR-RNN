import numpy as np

from . import info
from .data_loader import build_vocab


UNDEFINED = "U"
CHARACTER_TYPES = [
    ("N", "Half-width digits", 0x30, 0x39),
    ("N", "Full-width digits", 0xFF10, 0xFF19),
    ("R", "Half-width uppercase roman characters", 0x41, 0x5A),
    ("R", "Half-width lowercase roman characters", 0x61, 0x7A),
    ("R", "Full-width uppercase roman characters", 0xFF21, 0xFF3A),
    ("R", "Full-width lowercase roman characters", 0xFF41, 0xFF5A),
    ("ひ", "Hiragana", 0x3040, 0x3096),
    ("カ", "Katakana", 0x30A0, 0x30FF),
    (UNDEFINED, "Katakana middle dot", 0x30FB, 0x30FB),  # Treat as undefined
    ("カ", "Half-width katakana", 0xFF66, 0xFF9F),
    ("漢", "CJK Unified Ideographs", 0x4E00, 0x9FFC),
    ("漢", "CJK Unified Ideographs Extension A", 0x3400, 0x4DBF),
    ("漢", "CJK Unified Ideographs Extension B", 0x20000, 0x2A6DF),
    ("漢", "CJK Unified Ideographs Extension C", 0x2A700, 0x2B73F),
    ("漢", "CJK Unified Ideographs Extension D", 0x2B740, 0x2B81F),
    ("漢", "CJK Unified Ideographs Extension E", 0x2B820, 0x2CEAF),
    ("漢", "CJK Unified Ideographs Extension F", 0x2CEB0, 0x2EBE0),
    ("漢", "CJK Unified Ideographs Extension G", 0x30000, 0x3134A),
    ("漢", "CJK Compatibility Ideographs", 0xF900, 0xFAFF),
    ("漢", "CJK Compatibility Ideographs Supplement", 0x2F800, 0x2FA1F),
]


def build_code_point_array(definitions):
    """Build a code point array for extracting character type features.

    Args:
        definitions (list of tuple): Character type definitions.
            A definition is a tuple that consists of the following elements:
            - element#0 (str): Identifier. It must be a single character.
            - element#1 (str): User-friendly description.
            - element#2 (int): Start code point value.
            - element#3 (int): End code point value.
            The identifier element#0 is assigned to the code points from element#2 to element#3.
    Returns:
        list of str: Code point array built from the given definitions.
            Undefined code points are represented by 'U'.
    """
    CODE_POINT_MAX = 0x10FFFF
    cp_array = [UNDEFINED for _ in range(CODE_POINT_MAX + 1)]
    for identifier, _, start, end in definitions:
        for i in range(start, end + 1):
            cp_array[i] = identifier
    return cp_array


CODE_POINT_ARRAY = build_code_point_array(CHARACTER_TYPES)


def to_chartype_str(surface):
    """Convert a surface string to the character type string.

    Args:
        surface (str): Surface string.
    Returns:
        str: Character type string.
    """
    return "".join([CODE_POINT_ARRAY[ord(c)] for c in surface])


def extract_ngram_from_surface(surface, name, n_gram, window_size):
    """Extract character-level N-gram features from the given surface string.

    Args:
        surface (str): Surface string.
        name (str): Feature name. It is used as a prefix of the feature key.
        n_gram (int): N value of N-gram.
        window_size (int): Context window size.
    Returns:
        list of dict: Feature-value dicts.
    """
    if n_gram < 1:
        raise ValueError("n_gram must be a positive integer.")
    if window_size < 1:
        raise ValueError("window_size must be a positive integer.")

    feature_dicts = [None] * len(surface)
    for i in range(len(surface)):
        start = i - (window_size - 1)
        end = i + (window_size + 1) - (n_gram - 1)
        feature_dict = {}
        for window_idx, j in enumerate(range(start, end)):
            key = "{}_{}_{}".format(name, n_gram, window_idx)
            if j < 0 or j >= len(surface) - (n_gram - 1):
                feature_dict[key] = info.pad
            else:
                feature_dict[key] = surface[j : j + n_gram]
        feature_dicts[i] = feature_dict

    return feature_dicts


def extract_ngram_from_tokens(tokens, name, n_gram, window_size, position=None):
    """Extract character-level N-gram features from the given tokens.

    Args:
        tokens (list of str): Tokens.
        name (str): Feature name. It is used as a prefix of the feature key.
        n_gram (int): N value of N-gram.
        window_size (int): Context window size.
        position (int): Index of a token.
            If it is not None, the function only extracts features from the token at the position.
    Returns:
        list of dict: Feature-value dicts.
    """
    if n_gram < 1:
        raise ValueError("n_gram must be a positive integer.")
    if window_size < 1:
        raise ValueError("window_size must be a positive integer.")

    surface = "".join(tokens)
    token_start = 0
    token_end = 0
    feature_dicts = []

    for i, token in enumerate(tokens):
        token_start = token_end
        token_end = token_start + len(token)

        if position is not None and i != position:
            continue

        window = []
        for j in range(token_start - window_size, token_start):
            if j < 0:
                window.append("")
            else:
                window.append(surface[j])
        for j in range(token_end, token_end + window_size):
            if j >= len(surface):
                window.append("")
            else:
                window.append(surface[j])

        feature_dict = {}
        for window_idx in range(len(window) - (n_gram - 1)):
            key = "{}_{}_{}".format(name, n_gram, window_idx)
            str_ = "".join(window[window_idx : window_idx + n_gram])
            if len(str_) != n_gram:
                feature_dict[key] = info.pad
            else:
                feature_dict[key] = str_
        feature_dicts.append(feature_dict)

    return feature_dicts


CHARACTER_FEATURE_NAME = "char"
CHARACTER_TYPE_FEATURE_NAME = "chartype"
WORD_FEATURE_NAME = "word"


def extract_char_from_surface(surface, n_gram, window_size):
    return extract_ngram_from_surface(
        surface, CHARACTER_FEATURE_NAME, n_gram, window_size
    )


def extract_char_from_tokens(tokens, n_gram, window_size, position=None):
    return extract_ngram_from_tokens(
        tokens, CHARACTER_FEATURE_NAME, n_gram, window_size, position
    )


def extract_chartype_from_surface(surface, n_gram, window_size):
    return extract_ngram_from_surface(
        to_chartype_str(surface), CHARACTER_TYPE_FEATURE_NAME, n_gram, window_size
    )


def extract_chartype_from_tokens(tokens, n_gram, window_size, position=None):
    return extract_ngram_from_tokens(
        [to_chartype_str(t) for t in tokens],
        CHARACTER_TYPE_FEATURE_NAME,
        n_gram,
        window_size,
        position,
    )


def extract_word_features(tokens):
    """Extract Word features.

    Args:
        tokens (list of str): A list of tokens.
    Returns:
        list of dict: Feature-value dicts.
    """
    return [{WORD_FEATURE_NAME: token} for token in tokens]


def feature_dicts_to_indices(feature_dicts, ttoi):
    """Convert feature dicts into indices.

    Args:
        feature_dicts (list of dict): Feature-value dicts.
        ttoi (dict): A mapping from tokens to indices.
    Returns:
        np.ndarray, shape=(len(feature_dicts), values): Indices.
    """
    features = []
    unk_idx = ttoi[info.unk]

    for fd in feature_dicts:
        f = []
        for token in fd.values():
            index = ttoi.get(token)
            if index is None:
                f.append(unk_idx)
            else:
                f.append(index)
        features.append(f)

    return np.array(features, dtype=np.int64)


def extract_features(seqs, extract_fn, **kwargs):
    """Convinience function to extract features for recurrent neural networks.

    Args:
        seqs (list of str, list of list of str):
            List of unsegmented sentences or list of tokens.
        extract_fn (function): Feature extraction function.
    Returns:
        tuple[np.ndarray, dict]: Feature indices and token-to-index mapping.
    """
    feature_dicts = []
    for s in seqs:
        feature_dicts.extend(extract_fn(s, **kwargs))
    ttoi = build_vocab([v for d in feature_dicts for v in d.values()])
    features = feature_dicts_to_indices(feature_dicts, ttoi)
    return features, ttoi
