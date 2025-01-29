from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, BatchSampler, Sampler, TensorDataset

from . import info


class SequentialBatchSampler(Sampler):
    """Yield a minibatch of indices for sequential data.

    Args:
        indices (1-D np.ndarray): Sequence of indices.
        batch_size (int): Batch size.
    """

    def __init__(self, indices, batch_size):
        if not isinstance(indices, np.ndarray):
            raise ValueError(f"indices must be np.ndarray, but got {type(indices)}")
        if indices.ndim != 1:
            raise ValueError(f"indices.ndim must be 1, but got {indices.ndim}")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(
                f"batch_size must be a positive integer, but got {batch_size}"
            )
        batch_len = indices.shape[0] // batch_size
        if batch_len == 0:
            raise ValueError(
                f"batch_len must be a positive integer, but got {batch_len}"
            )
        self._indices = indices
        self._batch_size = batch_size
        self._batch_len = batch_len

    def __iter__(self):
        indices = self._indices[: self._batch_size * self._batch_len].reshape(
            self._batch_size, -1
        )
        for i in range(self._batch_len):
            yield [idx for idx in indices[:, i]]

    def __len__(self):
        return self._batch_len


class TimeStepSampler(BatchSampler):
    """Yield a time step of minibatches for Back-Propagation Through Time (BPTT).
    q
        Args:
            indices (1-D np.ndarray): Sequence of indices.
            batch_size (int): Batch size.
            unroll_size(int): Step size for BPTT.
    """

    def __init__(self, indices, batch_size, unroll_size):
        sampler = SequentialBatchSampler(indices, batch_size)
        super().__init__(sampler, unroll_size, True)


def build_data_loader(X, y, batch_size, unroll_size):
    """Build a data loader for training.

    Args:
        X (list of 2-D np.ndarray): Iterable of input indices.
        y (np.ndarray): Labels.
        batch_size (int): Batch size.
        unroll_size (int): Step size for Back-Propagation Through Time (BPTT).
    """
    for x in X:
        assert x.dtype == np.int64
    assert y.dtype == np.int64

    ds = TensorDataset(*[torch.from_numpy(x) for x in X], torch.from_numpy(y))
    bs = TimeStepSampler(np.arange(len(ds)), batch_size, unroll_size)
    return DataLoader(ds, batch_sampler=bs, pin_memory=True)


def tokenize(surface, y):
    """Tokenize a surface string with the given labels.

    Args:
        surface (str): Surface string.
        y (array_like, shape=(len(surface),)): Word boundary labels.
    Returns:
        list of str: Tokens.
    """
    if len(surface) != len(y):
        raise ValueError(
            "The length of the surface string and the label array must be the same."
        )

    tokens = []
    token = ""
    for c, label in zip(surface, y):
        token += c
        if label == 1:
            tokens.append(token)
            token = ""
    if token:
        tokens.append(token)

    return tokens


class TokenizerEvalDataset(Dataset):
    """
    Args:
        seqs (list of str): Surface strings.
        y (np.ndarray[np.int64]): Labels.
    """

    def __init__(self, seqs, y):
        self._seqs = seqs
        self._y = y
        self._indices = []

        assert sum([len(s) for s in self._seqs]) == len(self._y)
        start = 0
        for s in self._seqs:
            end = start + len(s)
            self._indices.append((start, end))
            start = end

    def __getitem__(self, index):
        start, end = self._indices[index]
        seq = self._seqs[index]
        return seq, tokenize(seq, self._y[start:end])

    def __len__(self):
        return len(self._seqs)


class TaggerEvalDataset(TokenizerEvalDataset):
    """
    Args:
        seqs (list of list of str): Lists of tokens.
        y (np.ndarray[np.int64]): Labels.
        itot (list of int, optional): Indices-to-tags mapping.
    """

    def __init__(self, seqs, y, itot):
        super().__init__(seqs, y)
        self._itot = itot

    def __getitem__(self, index):
        start, end = self._indices[index]
        seq = self._seqs[index]
        tags = [self._itot[i] for i in self._y[start:end]]
        return seq, tags


def load_corpus_data_for_word_segmentation(corpus):
    """Load data from the given corpus file for word segmentation.

    Args:
        corpus (str): Path to a corpus file.
    Returns:
        tuple[list of str, np.ndarray[np.int64]]:
            Surface strings and word boundary labels.
            Word boundary labels are {0, 1}.
    """
    unsegmented_sentences = []
    word_boundary_labels = []

    with open(corpus, encoding=info.charset) as f:
        for line in f:
            unsegmented_sentence = ""
            wb_labels = []
            for morph_str in line.replace(info.newline, "").split():
                token = morph_str.split(info.sep)[0]
                unsegmented_sentence += token
                label = [0 for _ in token]
                label[-1] = 1
                wb_labels.extend(label)
            unsegmented_sentences.append(unsegmented_sentence)
            word_boundary_labels.extend(wb_labels)

    return unsegmented_sentences, np.array(word_boundary_labels, dtype=np.int64)


def load_corpus_data_for_pos_tagging(corpus):
    """Load data from the given corpus file for Part-of-Speech tagging.

    Args:
        corpus (str): Path to a corpus file.
    Returns:
        tuple[list of list of str, list of list of str]:
            Sentences and list of tags. Each sentence is a list of tokens.
            len(tags) depends on the number of kinds of tags in the corpus.
    """
    sentences = []
    tags = None  # Instantiate later

    with open(corpus, encoding=info.charset) as f:
        for line in f:
            tokens = []
            for morph_str in line.replace(info.newline, "").split():
                morph = morph_str.split(info.sep)
                tokens.append(morph[0])

                if len(morph) == 1:
                    raise ValueError("Morpheme information is missing.")

                # Instantiate tags
                if tags is None:
                    tags = [[] for _ in range(len(morph) - 1)]

                for i, tag in enumerate(morph[1:]):
                    tags[i].append(tag)

            sentences.append(tokens)

    return sentences, tags


def build_vocab(tokens, min_count=5):
    """Build a map from tokens to indices.

    Args:
        tokens (list of str): List of tokens.
        min_count (int, optional): Minimum count of tokens to be included in the vocabulary.
    Returns:
        dict: A map from tokens to indices.
    """
    ttoi = {}
    counter = Counter(tokens)

    for token, count in counter.items():
        if min_count <= count:
            ttoi[token] = len(ttoi)

    # Add padding and unknown tokens
    if info.pad not in ttoi:
        ttoi[info.pad] = len(ttoi)
    if info.unk not in ttoi:
        ttoi[info.unk] = len(ttoi)

    return ttoi
