import numpy as np
import torch

from .feature_extraction import (
    extract_char_from_surface,
    extract_chartype_from_surface,
    feature_dicts_to_indices,
)
from . import data_loader


class LRTokenizer:
    """A logistic regression based tokenizer.

    Args:
        vectorizer (DictVectorizer): DictVectorizer instance.
        estimator (LogisticRegression): LogisticRegression instance.
    """

    def __init__(self, vectorizer, estimator):
        self._vectorizer = vectorizer
        self._estimator = estimator

    def predict(self, surface):
        """Predict a sequence of tokens.

        Args:
            surface (str): Surface string.
        Returns:
            list of str: Tokens.
        """
        X = self._vectorizer.transform(self.extract_features(surface))
        y = self._estimator.predict(X)
        return data_loader.tokenize(surface, y)

    def predict_log_proba(self, surface):
        """Predict log probability of word boundaries.

        Args:
            surface (str): Surface string.
        Returns:
            np.ndarray, shape (len(surface), 2): Log probability.
        """
        X = self._vectorizer.transform(self.extract_features(surface))
        return self._estimator.predict_log_proba(X)

    @staticmethod
    def extract_features(surface):
        """Extract features from a surface string.

        Args:
            surface (str): Surface string.
        Returns:
            list of dict: Feature-value dicts.
        """
        feature_dicts = []
        for c1, c2, c3, ct1, ct2, ct3 in zip(
            extract_char_from_surface(surface, 1, 3),
            extract_char_from_surface(surface, 2, 3),
            extract_char_from_surface(surface, 3, 3),
            extract_chartype_from_surface(surface, 1, 3),
            extract_chartype_from_surface(surface, 2, 3),
            extract_chartype_from_surface(surface, 3, 3),
        ):
            feature_dicts.append(dict(**c1, **c2, **c3, **ct1, **ct2, **ct3))
        return feature_dicts


class RNNTokenizer:
    """A recurent neural network based tokenizer.

    Args:
        char1_ttoi (dict): A mapping from char 1-gram to indices.
        char2_ttoi (dict): A mapping from char 2-gram to indices.
        char3_ttoi (dict): A mapping from char 3-gram to indices.
        chartype1_ttoi (dict): A mapping from chartype 1-gram to indices.
        chartype2_ttoi (dict): A mapping from chartype 2-gram to indices.
        chartype3_ttoi (dict): A mapping from chartype 3-gram to indices.
        model (WordSegmentation): WordSegmentation instance.
        device (torch.device): Device where `model` exists.
    """

    def __init__(
        self,
        char1_ttoi,
        char2_ttoi,
        char3_ttoi,
        chartype1_ttoi,
        chartype2_ttoi,
        chartype3_ttoi,
        model,
        device,
    ):
        self._char1_ttoi = char1_ttoi
        self._char2_ttoi = char2_ttoi
        self._char3_ttoi = char3_ttoi
        self._chartype1_ttoi = chartype1_ttoi
        self._chartype2_ttoi = chartype2_ttoi
        self._chartype3_ttoi = chartype3_ttoi
        self._model = model
        self._device = device

    def predict(self, surface):
        """Predict a sequence of tokens.

        Args:
            surface (str): Surface string.
        Returns:
            list of str: Tokens.
        """
        y = np.argmax(self.predict_log_proba(surface), axis=1)
        return data_loader.tokenize(surface, y)

    def predict_log_proba(self, surface):
        """Predict log probability of word boundaries.

        Args:
            surface (str): Surface string.
        Returns:
            np.ndarray, shape (len(surface), 2): Log probability.
        """
        (
            char1_features,
            char2_features,
            char3_features,
            chartype1_features,
            chartype2_features,
            chartype3_features,
        ) = self.extract_features(
            surface,
            self._char1_ttoi,
            self._char2_ttoi,
            self._char3_ttoi,
            self._chartype1_ttoi,
            self._chartype2_ttoi,
            self._chartype3_ttoi,
        )
        char1_features = torch.from_numpy(char1_features).unsqueeze(1)
        char1_features = char1_features.to(self._device)
        char2_features = torch.from_numpy(char2_features).unsqueeze(1)
        char2_features = char2_features.to(self._device)
        char3_features = torch.from_numpy(char3_features).unsqueeze(1)
        char3_features = char3_features.to(self._device)
        chartype1_features = torch.from_numpy(chartype1_features).unsqueeze(1)
        chartype1_features = chartype1_features.to(self._device)
        chartype2_features = torch.from_numpy(chartype2_features).unsqueeze(1)
        chartype2_features = chartype2_features.to(self._device)
        chartype3_features = torch.from_numpy(chartype3_features).unsqueeze(1)
        chartype3_features = chartype3_features.to(self._device)

        self._model.eval()
        with torch.no_grad():
            output, _ = self._model(
                char1_features,
                char2_features,
                char3_features,
                chartype1_features,
                chartype2_features,
                chartype3_features,
            )
            output = torch.nn.functional.log_softmax(output, dim=2)
            return output.squeeze(1).cpu().numpy()

    @staticmethod
    def extract_feature_dicts(surface):
        """Extract feature-value dicts from a surface string.

        Args:
            surface (str): Surface string.
        Returns:
            tuple[list of dict, list of dict, list of dict, list of dict,
                  list of dict, list of dict, list of dict]:
                char {1,2,3}-gram, chartype {1,2,3}-gram feature-value dicts.
        """
        return (
            extract_char_from_surface(surface, 1, 3),
            extract_char_from_surface(surface, 2, 3),
            extract_char_from_surface(surface, 3, 3),
            extract_chartype_from_surface(surface, 1, 3),
            extract_chartype_from_surface(surface, 2, 3),
            extract_chartype_from_surface(surface, 3, 3),
        )

    @staticmethod
    def extract_features(
        surface,
        char1_ttoi,
        char2_ttoi,
        char3_ttoi,
        chartype1_ttoi,
        chartype2_ttoi,
        chartype3_ttoi,
    ):
        """Extract indices of features from a surface string.

        Args:
            surface (str): Surface string.
        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray,
                  np.ndarray, np.ndarray, np.ndarray]: Indices of features.
        """
        (
            char1_feature_dicts,
            char2_feature_dicts,
            char3_feature_dicts,
            type1_feature_dicts,
            type2_feature_dicts,
            type3_feature_dicts,
        ) = RNNTokenizer.extract_feature_dicts(surface)
        return (
            feature_dicts_to_indices(char1_feature_dicts, char1_ttoi),
            feature_dicts_to_indices(char2_feature_dicts, char2_ttoi),
            feature_dicts_to_indices(char3_feature_dicts, char3_ttoi),
            feature_dicts_to_indices(type1_feature_dicts, chartype1_ttoi),
            feature_dicts_to_indices(type2_feature_dicts, chartype2_ttoi),
            feature_dicts_to_indices(type3_feature_dicts, chartype3_ttoi),
        )


class LRRNNTokenizer:
    """A linear interpolation of LRTokenizer and RNNTokenizer.

    Args:
        tokenizer1 (LRTokenizer): LRTokenizer instance.
        tokenizer2 (RNNTokenizer): RNNTokenizer instance.
        alpha (float): Alpha value.
    """

    def __init__(self, lr_tokenizer, rnn_tokenizer, alpha):
        self._lr_tokenizer = lr_tokenizer
        self._rnn_tokenizer = rnn_tokenizer
        self._alpha = alpha

    def predict(self, surface):
        """Predict a sequence of tokens.

        Args:
            surface (str): Surface string.
        Returns:
            list of str: Tokens.
        """
        y = [np.argmax(p) for p in self.predict_log_proba(surface)]
        return data_loader.tokenize(surface, y)

    def predict_log_proba(self, surface):
        """Predict log probability of word boundaries.

        Args:
            surface (str): Surface string.
        Returns:
            np.ndarray, shape (len(surface), 2): Log probability.
        """
        log_prob_lr = self._lr_tokenizer.predict_log_proba(surface)
        log_prob_rnn = self._rnn_tokenizer.predict_log_proba(surface)
        return log_prob_lr * (1 - self._alpha) + log_prob_rnn * self._alpha
