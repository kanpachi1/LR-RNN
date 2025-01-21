import unittest

import numpy as np
import pytest
import torch

from lrrnn import info
from lrrnn import data_loader
from lrrnn.data_loader import (
    SequentialBatchSampler,
    TimeStepSampler,
    TokenizerEvalDataset,
    TaggerEvalDataset,
)


class TestSequentialBatchSampler(unittest.TestCase):

    def test___init__(self):
        with self.assertRaises(ValueError) as cm:
            _ = SequentialBatchSampler([], None)
        self.assertEqual(
            str(cm.exception),
            "indices must be np.ndarray, but got <class 'list'>",
        )
        with self.assertRaises(ValueError) as cm:
            _ = SequentialBatchSampler(np.array([[]]), None)
        self.assertEqual(str(cm.exception), "indices.ndim must be 1, but got 2")
        with self.assertRaises(ValueError) as cm:
            _ = SequentialBatchSampler(np.array([]), 0)
        self.assertEqual(
            str(cm.exception),
            "batch_size must be a positive integer, but got 0",
        )
        with self.assertRaises(ValueError) as cm:
            _ = SequentialBatchSampler(np.arange(1), 2)
        self.assertEqual(
            str(cm.exception),
            "batch_len must be a positive integer, but got 0",
        )

    def test___iter__(self):
        indices = np.arange(9)
        batch_size = 2
        sampler = iter(SequentialBatchSampler(indices, batch_size))
        self.assertEqual(next(sampler), [0, 4])
        self.assertEqual(next(sampler), [1, 5])
        self.assertEqual(next(sampler), [2, 6])
        self.assertEqual(next(sampler), [3, 7])
        with self.assertRaises(StopIteration):
            _ = next(sampler)

    def test___len__(self):
        indices = np.arange(9)
        batch_size = 2
        self.assertEqual(len(SequentialBatchSampler(indices, batch_size)), 4)


class TestTimeStepSampler(unittest.TestCase):

    def test___iter__(self):
        indices = np.arange(10)
        batch_size = 2
        unroll_size = 2
        sampler = iter(TimeStepSampler(indices, batch_size, unroll_size))
        self.assertEqual(next(sampler), [[0, 5], [1, 6]])
        self.assertEqual(next(sampler), [[2, 7], [3, 8]])
        with self.assertRaises(StopIteration):
            _ = next(sampler)

    def test___len__(self):
        indices = np.arange(10)
        batch_size = 2
        unroll_size = 2
        self.assertEqual(len(TimeStepSampler(indices, batch_size, unroll_size)), 2)


class TestTokenizerEvalDataset(unittest.TestCase):
    def test_surface_strings(self):
        strs = ["surfacestrings0", "surfacestrings1"]
        ys = np.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
            ],
            dtype=np.int64,
        )
        ds = TokenizerEvalDataset(strs, ys)
        self.assertEqual(len(ds), 2)
        s, y = ds[0]
        self.assertEqual(s, strs[0])
        self.assertEqual(y, ["surface", "strings", "0"])
        s, y = ds[1]
        self.assertEqual(s, strs[1])
        self.assertEqual(y, ["surface", "strings", "1"])

        dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
        self.assertEqual(len(dl), 2)
        it = iter(dl)
        self.assertEqual(next(it), [(strs[0],), [("surface",), ("strings",), ("0",)]])
        self.assertEqual(next(it), [(strs[1],), [("surface",), ("strings",), ("1",)]])
        with self.assertRaises(StopIteration):
            _ = next(it)


class TestTaggerEvalDataset(unittest.TestCase):
    def test_tokens(self):
        tokens = [["tokens", "0"], ["tokens", "1"]]
        ys = np.array([0, 1, 0, 2], dtype=np.int64)
        itot = ["noun", "number_0", "number_1"]
        ds = TaggerEvalDataset(tokens, ys, itot)
        self.assertEqual(len(ds), 2)
        s, y = ds[0]
        self.assertEqual(s, tokens[0])
        self.assertEqual(y, ["noun", "number_0"])
        s, y = ds[1]
        self.assertEqual(s, tokens[1])
        self.assertEqual(y, ["noun", "number_1"])

        dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
        self.assertEqual(len(dl), 2)
        it = iter(dl)
        self.assertEqual(next(it), [[("tokens",), ("0",)], [("noun",), ("number_0",)]])
        self.assertEqual(next(it), [[("tokens",), ("1",)], [("noun",), ("number_1",)]])
        with self.assertRaises(StopIteration):
            _ = next(it)


def test_build_data_loader():
    X = [
        np.arange(60, dtype=np.int64).reshape(10, 6),
        np.arange(50, dtype=np.int64).reshape(10, 5),
        np.arange(40, dtype=np.int64).reshape(10, 4),
    ]
    y = np.arange(10)
    batch_size = 2
    unroll_size = 2
    dl = iter(data_loader.build_data_loader(X, y, batch_size, unroll_size))
    x1b, x2b, x3b, yb = next(dl)
    assert torch.equal(
        x1b,
        torch.tensor(
            [
                [[0, 1, 2, 3, 4, 5], [30, 31, 32, 33, 34, 35]],
                [[6, 7, 8, 9, 10, 11], [36, 37, 38, 39, 40, 41]],
            ]
        ),
    )
    assert torch.equal(
        x2b,
        torch.tensor(
            [
                [[0, 1, 2, 3, 4], [25, 26, 27, 28, 29]],
                [[5, 6, 7, 8, 9], [30, 31, 32, 33, 34]],
            ]
        ),
    )
    assert torch.equal(
        x3b,
        torch.tensor(
            [[[0, 1, 2, 3], [20, 21, 22, 23]], [[4, 5, 6, 7], [24, 25, 26, 27]]]
        ),
    )
    assert torch.equal(yb, torch.tensor([[0, 5], [1, 6]]))

    x1b, x2b, x3b, yb = next(dl)
    assert torch.equal(
        x1b,
        torch.tensor(
            [
                [[12, 13, 14, 15, 16, 17], [42, 43, 44, 45, 46, 47]],
                [[18, 19, 20, 21, 22, 23], [48, 49, 50, 51, 52, 53]],
            ]
        ),
    )
    assert torch.equal(
        x2b,
        torch.tensor(
            [
                [[10, 11, 12, 13, 14], [35, 36, 37, 38, 39]],
                [[15, 16, 17, 18, 19], [40, 41, 42, 43, 44]],
            ]
        ),
    )
    assert torch.equal(
        x3b,
        torch.tensor(
            [
                [[8, 9, 10, 11], [28, 29, 30, 31]],
                [[12, 13, 14, 15], [32, 33, 34, 35]],
            ]
        ),
    )
    assert torch.equal(yb, torch.tensor([[2, 7], [3, 8]]))
    with pytest.raises(StopIteration):
        _ = next(dl)


def test_build_vocab():
    tokens = [
        "foo",
        "bar",
        "hoge",
        "hoge",
        "hoge",
        "hoge",
        "hoge",
        info.pad,
        info.unk,
    ]
    ttoi = data_loader.build_vocab(tokens)
    assert ttoi == {"hoge": 0, info.pad: 1, info.unk: 2}
    ttoi = data_loader.build_vocab(tokens, min_count=1)
    assert ttoi == {"foo": 0, "bar": 1, "hoge": 2, info.pad: 3, info.unk: 4}


def test_tokenize():
    surface = "helloworld"
    assert data_loader.tokenize(surface, [0, 0, 0, 0, 1, 0, 0, 0, 0, 1]) == [
        "hello",
        "world",
    ]
    assert data_loader.tokenize(surface, [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]) == [
        "hello",
        "world",
    ]


def test_load_corpus_data_for_word_segmentation(tmp_path):
    # Corpus file with surface, tag0 and tag1
    corpus_file = tmp_path / "corpus.txt"
    corpus_file.write_text(
        (
            "hello/tag00/tag10 world/tag01/tag11\n"
            "HELLO/tag02/tag12 WORLD/tag03/tag13\n"
        )
    )
    unsegmented_sentences, word_boundary_labels = (
        data_loader.load_corpus_data_for_word_segmentation(corpus_file)
    )

    assert unsegmented_sentences == ["helloworld", "HELLOWORLD"]
    assert word_boundary_labels.dtype == np.int64
    assert np.array_equal(
        word_boundary_labels,
        np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
    )


def test_load_corpus_data_for_word_segmentation_notag(tmp_path):
    # Corpus file containing only surface strings, without tags
    corpus_notags = tmp_path / "corpus_notags.txt"
    corpus_notags.write_text("hello world\nHELLO WORLD\n")
    unsegmented_sentences, word_boundary_labels = (
        data_loader.load_corpus_data_for_word_segmentation(corpus_notags)
    )

    assert unsegmented_sentences == ["helloworld", "HELLOWORLD"]
    assert word_boundary_labels.dtype == np.int64
    assert np.array_equal(
        word_boundary_labels,
        np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
    )


def test_load_corpus_data_for_pos_tagging(tmp_path):
    # Corpus file with surface, tag0 and tag1
    corpus_file = tmp_path / "corpus.txt"
    corpus_file.write_text(
        (
            "hello/tag00/tag10 world/tag01/tag11\n"
            "HELLO/tag02/tag12 WORLD/tag03/tag13\n"
        )
    )
    sentences, tags = data_loader.load_corpus_data_for_pos_tagging(corpus_file)

    assert sentences == [["hello", "world"], ["HELLO", "WORLD"]]
    assert tags == [
        ["tag00", "tag01", "tag02", "tag03"],
        ["tag10", "tag11", "tag12", "tag13"],
    ]


def test_load_corpus_data_for_pos_tagging_notag(tmp_path):
    # Corpus file containing only surface strings, without tags
    corpus_notags = tmp_path / "corpus_notags.txt"
    corpus_notags.write_text("hello world\nHELLO WORLD\n")

    with pytest.raises(ValueError) as cm:
        _ = data_loader.load_corpus_data_for_pos_tagging(corpus_notags)
