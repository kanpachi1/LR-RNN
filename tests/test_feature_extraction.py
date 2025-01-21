import numpy as np
import pytest

from lrrnn import info
from lrrnn.feature_extraction import (
    UNDEFINED,
    CHARACTER_TYPES,
    build_code_point_array,
    to_chartype_str,
    extract_ngram_from_surface,
    extract_ngram_from_tokens,
    extract_word_features,
    feature_dicts_to_indices,
)


def test_build_code_point_array():
    code_point_array = build_code_point_array(CHARACTER_TYPES)
    assert len(code_point_array) == 0x110000

    def boundary_check(cp, id0, id1):
        if isinstance(cp, str):
            cp = ord(cp[0])
        assert code_point_array[cp] == id0
        assert code_point_array[cp + 1] == id1

    # Undefined
    assert code_point_array[0] == UNDEFINED
    assert code_point_array[0x10FF] == UNDEFINED
    # Digits
    boundary_check(0x30 - 1, UNDEFINED, "N")
    boundary_check(0x39, "N", UNDEFINED)
    boundary_check(0xFF10 - 1, UNDEFINED, "N")
    boundary_check(0xFF19, "N", UNDEFINED)
    # Roman characters
    boundary_check(0x41 - 1, UNDEFINED, "R")
    boundary_check(0x5A, "R", UNDEFINED)
    boundary_check(0x61 - 1, UNDEFINED, "R")
    boundary_check(0x7A, "R", UNDEFINED)
    boundary_check(0xFF21 - 1, UNDEFINED, "R")
    boundary_check(0xFF3A, "R", UNDEFINED)
    boundary_check(0xFF41 - 1, UNDEFINED, "R")
    boundary_check(0xFF5A, "R", UNDEFINED)
    # Hiragana
    boundary_check(0x3040 - 1, UNDEFINED, "ひ")
    boundary_check(0x3096, "ひ", UNDEFINED)
    # Katakana
    boundary_check(0x30A0 - 1, UNDEFINED, "カ")
    boundary_check(0x30FF, "カ", UNDEFINED)
    boundary_check(0x30FB - 1, "カ", UNDEFINED)
    boundary_check(0x30FB, UNDEFINED, "カ")
    boundary_check(0xFF66 - 1, UNDEFINED, "カ")
    boundary_check(0xFF9F, "カ", UNDEFINED)
    # Kanji
    boundary_check(0x4E00 - 1, UNDEFINED, "漢")
    boundary_check(0x9FFC, "漢", UNDEFINED)
    boundary_check(0x3400 - 1, UNDEFINED, "漢")
    boundary_check(0x4DBF, "漢", UNDEFINED)
    boundary_check(0x20000 - 1, UNDEFINED, "漢")
    boundary_check(0x2A6DF, "漢", UNDEFINED)
    boundary_check(0x2A700 - 1, UNDEFINED, "漢")
    boundary_check(0x2B73F, "漢", "漢")
    boundary_check(0x2B740 - 1, "漢", "漢")
    boundary_check(0x2B81F, "漢", "漢")
    boundary_check(0x2B820 - 1, "漢", "漢")
    boundary_check(0x2CEAF, "漢", "漢")
    boundary_check(0x2CEB0 - 1, "漢", "漢")
    boundary_check(0x2EBE0, "漢", UNDEFINED)
    boundary_check(0x30000 - 1, UNDEFINED, "漢")
    boundary_check(0x3134A, "漢", UNDEFINED)
    boundary_check(0xF900 - 1, UNDEFINED, "漢")
    boundary_check(0xFAFF, "漢", UNDEFINED)
    boundary_check(0x2F800 - 1, UNDEFINED, "漢")
    boundary_check(0x2FA1F, "漢", UNDEFINED)


def test_to_chartype_str():
    assert to_chartype_str("CharType0") == "RRRRRRRRN"
    assert to_chartype_str("ＣｈａｒＴｙｐｅ０") == "RRRRRRRRN"
    assert to_chartype_str("犬も歩けば棒に当たる。") == "漢ひ漢ひひ漢ひ漢ひひU"
    assert to_chartype_str("えびフライ") == "ひひカカカ"


def test_extract_ngram_from_surface_unigram():
    assert extract_ngram_from_surface("abcdef", "unigram", 1, 3) == [
        {
            ("unigram_1_0"): info.pad,
            ("unigram_1_1"): info.pad,
            ("unigram_1_2"): "a",
            ("unigram_1_3"): "b",
            ("unigram_1_4"): "c",
            ("unigram_1_5"): "d",
        },
        {
            ("unigram_1_0"): info.pad,
            ("unigram_1_1"): "a",
            ("unigram_1_2"): "b",
            ("unigram_1_3"): "c",
            ("unigram_1_4"): "d",
            ("unigram_1_5"): "e",
        },
        {
            ("unigram_1_0"): "a",
            ("unigram_1_1"): "b",
            ("unigram_1_2"): "c",
            ("unigram_1_3"): "d",
            ("unigram_1_4"): "e",
            ("unigram_1_5"): "f",
        },
        {
            ("unigram_1_0"): "b",
            ("unigram_1_1"): "c",
            ("unigram_1_2"): "d",
            ("unigram_1_3"): "e",
            ("unigram_1_4"): "f",
            ("unigram_1_5"): info.pad,
        },
        {
            ("unigram_1_0"): "c",
            ("unigram_1_1"): "d",
            ("unigram_1_2"): "e",
            ("unigram_1_3"): "f",
            ("unigram_1_4"): info.pad,
            ("unigram_1_5"): info.pad,
        },
        {
            ("unigram_1_0"): "d",
            ("unigram_1_1"): "e",
            ("unigram_1_2"): "f",
            ("unigram_1_3"): info.pad,
            ("unigram_1_4"): info.pad,
            ("unigram_1_5"): info.pad,
        },
    ]


def test_extract_ngram_from_surface_bigram():
    assert extract_ngram_from_surface("abcdef", "bigram", 2, 3) == [
        {
            ("bigram_2_0"): info.pad,
            ("bigram_2_1"): info.pad,
            ("bigram_2_2"): "ab",
            ("bigram_2_3"): "bc",
            ("bigram_2_4"): "cd",
        },
        {
            ("bigram_2_0"): info.pad,
            ("bigram_2_1"): "ab",
            ("bigram_2_2"): "bc",
            ("bigram_2_3"): "cd",
            ("bigram_2_4"): "de",
        },
        {
            ("bigram_2_0"): "ab",
            ("bigram_2_1"): "bc",
            ("bigram_2_2"): "cd",
            ("bigram_2_3"): "de",
            ("bigram_2_4"): "ef",
        },
        {
            ("bigram_2_0"): "bc",
            ("bigram_2_1"): "cd",
            ("bigram_2_2"): "de",
            ("bigram_2_3"): "ef",
            ("bigram_2_4"): info.pad,
        },
        {
            ("bigram_2_0"): "cd",
            ("bigram_2_1"): "de",
            ("bigram_2_2"): "ef",
            ("bigram_2_3"): info.pad,
            ("bigram_2_4"): info.pad,
        },
        {
            ("bigram_2_0"): "de",
            ("bigram_2_1"): "ef",
            ("bigram_2_2"): info.pad,
            ("bigram_2_3"): info.pad,
            ("bigram_2_4"): info.pad,
        },
    ]


def test_extract_ngram_from_surface_trigram():
    assert extract_ngram_from_surface("abcdef", "trigram", 3, 3) == [
        {
            ("trigram_3_0"): info.pad,
            ("trigram_3_1"): info.pad,
            ("trigram_3_2"): "abc",
            ("trigram_3_3"): "bcd",
        },
        {
            ("trigram_3_0"): info.pad,
            ("trigram_3_1"): "abc",
            ("trigram_3_2"): "bcd",
            ("trigram_3_3"): "cde",
        },
        {
            ("trigram_3_0"): "abc",
            ("trigram_3_1"): "bcd",
            ("trigram_3_2"): "cde",
            ("trigram_3_3"): "def",
        },
        {
            ("trigram_3_0"): "bcd",
            ("trigram_3_1"): "cde",
            ("trigram_3_2"): "def",
            ("trigram_3_3"): info.pad,
        },
        {
            ("trigram_3_0"): "cde",
            ("trigram_3_1"): "def",
            ("trigram_3_2"): info.pad,
            ("trigram_3_3"): info.pad,
        },
        {
            ("trigram_3_0"): "def",
            ("trigram_3_1"): info.pad,
            ("trigram_3_2"): info.pad,
            ("trigram_3_3"): info.pad,
        },
    ]


def test_extract_ngram_from_surface_invalid():
    surface = "abcdef"

    with pytest.raises(ValueError) as cm:
        _ = extract_ngram_from_surface(surface, "invalid", 0, 3)
        assert cm.value.args[0] == "n_gram must be a positive integer."

    with pytest.raises(ValueError) as cm:
        _ = extract_ngram_from_surface(surface, "invalid", 1, 0)
        assert cm.value.args[0] == "window_size must be a positive integer."


def test_extract_ngram_from_tokens_unigram():
    tokens = ["A", "dog", "runs", "fast", "."]
    assert extract_ngram_from_tokens(tokens, "unigram", 1, 3) == [
        {
            ("unigram_1_0"): info.pad,
            ("unigram_1_1"): info.pad,
            ("unigram_1_2"): info.pad,
            ("unigram_1_3"): "d",
            ("unigram_1_4"): "o",
            ("unigram_1_5"): "g",
        },
        {
            ("unigram_1_0"): info.pad,
            ("unigram_1_1"): info.pad,
            ("unigram_1_2"): "A",
            ("unigram_1_3"): "r",
            ("unigram_1_4"): "u",
            ("unigram_1_5"): "n",
        },
        {
            ("unigram_1_0"): "d",
            ("unigram_1_1"): "o",
            ("unigram_1_2"): "g",
            ("unigram_1_3"): "f",
            ("unigram_1_4"): "a",
            ("unigram_1_5"): "s",
        },
        {
            ("unigram_1_0"): "u",
            ("unigram_1_1"): "n",
            ("unigram_1_2"): "s",
            ("unigram_1_3"): ".",
            ("unigram_1_4"): info.pad,
            ("unigram_1_5"): info.pad,
        },
        {
            ("unigram_1_0"): "a",
            ("unigram_1_1"): "s",
            ("unigram_1_2"): "t",
            ("unigram_1_3"): info.pad,
            ("unigram_1_4"): info.pad,
            ("unigram_1_5"): info.pad,
        },
    ]


def test_extract_ngram_from_tokens_unigram_with_position():
    tokens = ["A", "dog", "runs", "fast", "."]
    assert extract_ngram_from_tokens(tokens, "unigram", 1, 3, 0) == [
        {
            ("unigram_1_0"): info.pad,
            ("unigram_1_1"): info.pad,
            ("unigram_1_2"): info.pad,
            ("unigram_1_3"): "d",
            ("unigram_1_4"): "o",
            ("unigram_1_5"): "g",
        }
    ]
    assert extract_ngram_from_tokens(tokens, "unigram", 1, 3, 1) == [
        {
            ("unigram_1_0"): info.pad,
            ("unigram_1_1"): info.pad,
            ("unigram_1_2"): "A",
            ("unigram_1_3"): "r",
            ("unigram_1_4"): "u",
            ("unigram_1_5"): "n",
        }
    ]
    assert extract_ngram_from_tokens(tokens, "unigram", 1, 3, 2) == [
        {
            ("unigram_1_0"): "d",
            ("unigram_1_1"): "o",
            ("unigram_1_2"): "g",
            ("unigram_1_3"): "f",
            ("unigram_1_4"): "a",
            ("unigram_1_5"): "s",
        }
    ]
    assert extract_ngram_from_tokens(tokens, "unigram", 1, 3, 3) == [
        {
            ("unigram_1_0"): "u",
            ("unigram_1_1"): "n",
            ("unigram_1_2"): "s",
            ("unigram_1_3"): ".",
            ("unigram_1_4"): info.pad,
            ("unigram_1_5"): info.pad,
        }
    ]
    assert extract_ngram_from_tokens(tokens, "unigram", 1, 3, 4) == [
        {
            ("unigram_1_0"): "a",
            ("unigram_1_1"): "s",
            ("unigram_1_2"): "t",
            ("unigram_1_3"): info.pad,
            ("unigram_1_4"): info.pad,
            ("unigram_1_5"): info.pad,
        }
    ]


def test_extract_ngram_from_tokens_bigram():
    tokens = ["A", "dog", "runs", "fast", "."]
    assert extract_ngram_from_tokens(tokens, "bigram", 2, 3) == [
        {
            ("bigram_2_0"): info.pad,
            ("bigram_2_1"): info.pad,
            ("bigram_2_2"): info.pad,
            ("bigram_2_3"): "do",
            ("bigram_2_4"): "og",
        },
        {
            ("bigram_2_0"): info.pad,
            ("bigram_2_1"): info.pad,
            ("bigram_2_2"): "Ar",
            ("bigram_2_3"): "ru",
            ("bigram_2_4"): "un",
        },
        {
            ("bigram_2_0"): "do",
            ("bigram_2_1"): "og",
            ("bigram_2_2"): "gf",
            ("bigram_2_3"): "fa",
            ("bigram_2_4"): "as",
        },
        {
            ("bigram_2_0"): "un",
            ("bigram_2_1"): "ns",
            ("bigram_2_2"): "s.",
            ("bigram_2_3"): info.pad,
            ("bigram_2_4"): info.pad,
        },
        {
            ("bigram_2_0"): "as",
            ("bigram_2_1"): "st",
            ("bigram_2_2"): info.pad,
            ("bigram_2_3"): info.pad,
            ("bigram_2_4"): info.pad,
        },
    ]


def test_extract_ngram_from_tokens_bigram_with_position():
    tokens = ["A", "dog", "runs", "fast", "."]
    assert extract_ngram_from_tokens(tokens, "bigram", 2, 3, 0) == [
        {
            ("bigram_2_0"): info.pad,
            ("bigram_2_1"): info.pad,
            ("bigram_2_2"): info.pad,
            ("bigram_2_3"): "do",
            ("bigram_2_4"): "og",
        }
    ]
    assert extract_ngram_from_tokens(tokens, "bigram", 2, 3, 1) == [
        {
            ("bigram_2_0"): info.pad,
            ("bigram_2_1"): info.pad,
            ("bigram_2_2"): "Ar",
            ("bigram_2_3"): "ru",
            ("bigram_2_4"): "un",
        }
    ]
    assert extract_ngram_from_tokens(tokens, "bigram", 2, 3, 2) == [
        {
            ("bigram_2_0"): "do",
            ("bigram_2_1"): "og",
            ("bigram_2_2"): "gf",
            ("bigram_2_3"): "fa",
            ("bigram_2_4"): "as",
        }
    ]
    assert extract_ngram_from_tokens(tokens, "bigram", 2, 3, 3) == [
        {
            ("bigram_2_0"): "un",
            ("bigram_2_1"): "ns",
            ("bigram_2_2"): "s.",
            ("bigram_2_3"): info.pad,
            ("bigram_2_4"): info.pad,
        }
    ]
    assert extract_ngram_from_tokens(tokens, "bigram", 2, 3, 4) == [
        {
            ("bigram_2_0"): "as",
            ("bigram_2_1"): "st",
            ("bigram_2_2"): info.pad,
            ("bigram_2_3"): info.pad,
            ("bigram_2_4"): info.pad,
        }
    ]


def test_extract_ngram_from_tokens_trigram():
    tokens = ["A", "dog", "runs", "fast", "."]
    assert extract_ngram_from_tokens(tokens, "trigram", 3, 3) == [
        {
            ("trigram_3_0"): info.pad,
            ("trigram_3_1"): info.pad,
            ("trigram_3_2"): info.pad,
            ("trigram_3_3"): "dog",
        },
        {
            ("trigram_3_0"): info.pad,
            ("trigram_3_1"): info.pad,
            ("trigram_3_2"): "Aru",
            ("trigram_3_3"): "run",
        },
        {
            ("trigram_3_0"): "dog",
            ("trigram_3_1"): "ogf",
            ("trigram_3_2"): "gfa",
            ("trigram_3_3"): "fas",
        },
        {
            ("trigram_3_0"): "uns",
            ("trigram_3_1"): "ns.",
            ("trigram_3_2"): info.pad,
            ("trigram_3_3"): info.pad,
        },
        {
            ("trigram_3_0"): "ast",
            ("trigram_3_1"): info.pad,
            ("trigram_3_2"): info.pad,
            ("trigram_3_3"): info.pad,
        },
    ]


def test_extract_ngram_from_tokens_trigram_with_position():
    tokens = ["A", "dog", "runs", "fast", "."]
    assert extract_ngram_from_tokens(tokens, "trigram", 3, 3, 0) == [
        {
            ("trigram_3_0"): info.pad,
            ("trigram_3_1"): info.pad,
            ("trigram_3_2"): info.pad,
            ("trigram_3_3"): "dog",
        }
    ]
    assert extract_ngram_from_tokens(tokens, "trigram", 3, 3, 1) == [
        {
            ("trigram_3_0"): info.pad,
            ("trigram_3_1"): info.pad,
            ("trigram_3_2"): "Aru",
            ("trigram_3_3"): "run",
        }
    ]
    assert extract_ngram_from_tokens(tokens, "trigram", 3, 3, 2) == [
        {
            ("trigram_3_0"): "dog",
            ("trigram_3_1"): "ogf",
            ("trigram_3_2"): "gfa",
            ("trigram_3_3"): "fas",
        }
    ]
    assert extract_ngram_from_tokens(tokens, "trigram", 3, 3, 3) == [
        {
            ("trigram_3_0"): "uns",
            ("trigram_3_1"): "ns.",
            ("trigram_3_2"): info.pad,
            ("trigram_3_3"): info.pad,
        }
    ]
    assert extract_ngram_from_tokens(tokens, "trigram", 3, 3, 4) == [
        {
            ("trigram_3_0"): "ast",
            ("trigram_3_1"): info.pad,
            ("trigram_3_2"): info.pad,
            ("trigram_3_3"): info.pad,
        }
    ]


def test_extract_ngram_from_tokens_invalid():
    tokens = ["A", "dog", "runs", "fast", "."]

    with pytest.raises(ValueError) as cm:
        _ = extract_ngram_from_tokens(tokens, "invalid", 0, 3)
        assert cm.value.args[0] == "n_gram must be a positive integer."

    with pytest.raises(ValueError) as cm:
        _ = extract_ngram_from_tokens(tokens, "invalid", 1, 0)
        assert cm.value.args[0] == "window_size must be a positive integer."


def test_extract_word_features():
    tokens = ["Constant", "dripping", "wears", "away", "the", "stone", "."]
    assert extract_word_features(tokens) == [
        {"word": "Constant"},
        {"word": "dripping"},
        {"word": "wears"},
        {"word": "away"},
        {"word": "the"},
        {"word": "stone"},
        {"word": "."},
    ]


def test_feature_dicts_to_indices():
    feature_dicts = [
        {
            ("char_3_0"): info.pad,
            ("char_3_1"): info.pad,
            ("char_3_2"): "abc",
            ("char_3_3"): "bcd",
        },
        {
            ("char_3_0"): info.pad,
            ("char_3_1"): "abc",
            ("char_3_2"): "bcd",
            ("char_3_3"): "cde",
        },
    ]
    ttoi = {"abc": 0, "bcd": 1, info.pad: 2, info.unk: 3}
    assert np.array_equal(
        feature_dicts_to_indices(feature_dicts, ttoi),
        np.array([[2, 2, 0, 1], [2, 0, 1, 3]], dtype=np.int64),
    )
