import numpy as np
from scipy.special import softmax
import torch

from .feature_extraction import (
    extract_char_from_tokens,
    extract_chartype_from_tokens,
    extract_word_features,
    feature_dicts_to_indices,
)


class LRTagger:
    """A logistic regression based tagger.

    Args:
        estimators (dict): A map from tokens to LogisticRegression instances.
        vectorizers (dict): A map from tokens to DictVectorizer instances.
        ttois (dict): A map from tag strings to class labels.
        itots (list of str): A map from class labels to tag strings.
    """

    def __init__(self, estimators, vectorizer, ttois, itots):
        self._estimators = estimators
        self._vectorizer = vectorizer
        self._ttois = ttois
        self._itots = itots

    def predict(self, tokens):
        """Predict tags.

        Args:
            tokens (list of str): Tokens.
        Returns:
            list of str: Tags. If a token is OOV, its tag will be 'UNK'.
        """
        tags = []
        for i, token in enumerate(tokens):
            estimator = self._estimators.get(token, 0)
            if estimator == 0:
                tags.append("UNK")
            elif estimator is None:
                tags.append(self._itots[token][0])
            else:
                vectorizer = self._vectorizer.get(token)
                y = estimator.predict(
                    vectorizer.transform(self.extract_features(tokens, i))
                )
                tags.append(self._itots[token][y[0]])

        return tags

    def predict_single_log_proba(self, tokens, i):
        """Predict log probability of the token at the given index.

        Args:
            tokens (list of str): Tokens.
            i (int): Index of tokens.
        Returns:
            tuple (np.ndarray, dict, dict): Log probability, itot and ttoi.
                Log probability will be np.ones((1, 1)) if the token has only
                one class or is out of vocabulary.
        """
        estimator = self._estimators.get(tokens[i], 0)
        if estimator == 0:
            return (np.ones((1, 1)), ["UNK"], {"UNK": 0})
        else:
            itot = self._itots.get(tokens[i])
            ttoi = self._ttois.get(tokens[i])
            if itot is None:
                raise ValueError("itots are corrupted.")
            if ttoi is None:
                raise ValueError("ttois are corrupted.")
            if estimator is None:
                return (np.ones((1, 1)), itot, ttoi)
            else:
                vectorizer = self._vectorizer.get(tokens[i])
                if vectorizer is None:
                    raise ValueError("vectorizers are corrupted.")
                X = vectorizer.transform(self.extract_features(tokens, i))
                return (estimator.predict_log_proba(X), itot, ttoi)

    @staticmethod
    def extract_features(tokens, i):
        """Extract features at the given index of tokens.

        Args:
            tokens (list of str): Tokens.
            i (int): Index of a token.
        Returns:
            dict: Feature-value dict.
        """
        return dict(
            **extract_char_from_tokens(tokens, 1, 3, i)[0],
            **extract_char_from_tokens(tokens, 2, 3, i)[0],
            **extract_char_from_tokens(tokens, 3, 3, i)[0],
            **extract_chartype_from_tokens(tokens, 1, 3, i)[0],
            **extract_chartype_from_tokens(tokens, 2, 3, i)[0],
            **extract_chartype_from_tokens(tokens, 3, 3, i)[0],
        )


class RNNTagger:
    """A recurrent neural network based tagger.

    Args:
        char1_ttoi (dict): A mapping from char 1-gram to indices.
        char2_ttoi (dict): A mapping from char 2-gram to indices.
        char3_ttoi (dict): A mapping from char 3-gram to indices.
        chartype1_ttoi (dict): A mapping from chartype 1-gram to indices.
        chartype2_ttoi (dict): A mapping from chartype 2-gram to indices.
        chartype3_ttoi (dict): A mapping from chartype 3-gram to indices.
        word_ttoi (dict): A mapping from tokens to indices.
        model (POSTagging): POSTagging instance.
        device (torch.device): Device where `model` exists.
        tag_to_id (dict): A mapping from tag strings to class labels.
        id_to_tag (list): A mapping from class labels to tag strings.
    """

    def __init__(
        self,
        char1_ttoi,
        char2_ttoi,
        char3_ttoi,
        chartype1_ttoi,
        chartype2_ttoi,
        chartype3_ttoi,
        word_ttoi,
        model,
        device,
        tag_to_id,
        id_to_tag,
    ):
        self._char1_ttoi = char1_ttoi
        self._char2_ttoi = char2_ttoi
        self._char3_ttoi = char3_ttoi
        self._chartype1_ttoi = chartype1_ttoi
        self._chartype2_ttoi = chartype2_ttoi
        self._chartype3_ttoi = chartype3_ttoi
        self._word_ttoi = word_ttoi
        self._model = model
        self._device = device
        self._tag_to_id = tag_to_id
        self._id_to_tag = id_to_tag

    def predict(self, tokens):
        """Predict tags.

        Args:
            tokens (list of str): Tokens.
        Returns:
            list of str: Predicted tags.
        """
        y = np.argmax(self.predict_log_proba(tokens), axis=1)
        return [self._id_to_tag[i] for i in y]

    def predict_log_proba(self, tokens):
        """Predict log probability of tags.

        Args:
            tokens (list of str): Tokens
        Returns:
            np.ndarray, shape=(len(tokens), classes):
                Predicted log probability of tags.
        """
        (
            char1_features,
            char2_features,
            char3_features,
            chartype1_features,
            chartype2_features,
            chartype3_features,
            word_features,
        ) = self.extract_features(
            tokens,
            self._char1_ttoi,
            self._char2_ttoi,
            self._char3_ttoi,
            self._chartype1_ttoi,
            self._chartype2_ttoi,
            self._chartype3_ttoi,
            self._word_ttoi,
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
        word_features = torch.from_numpy(word_features).unsqueeze(1)
        word_features = word_features.to(self._device)

        self._model.eval()
        with torch.no_grad():
            output, _ = self._model(
                char1_features,
                char2_features,
                char3_features,
                chartype1_features,
                chartype2_features,
                chartype3_features,
                word_features,
            )
            output = torch.nn.functional.log_softmax(output, dim=2)
            return output.squeeze(1).cpu().numpy()

    @property
    def ttoi(self):
        return self._tag_to_id

    @staticmethod
    def extract_feature_dicts(tokens):
        """Extract feature-value dicts.

        Args:
            tokens (list of str): Tokens.
        Returns:
            tuple[list of dict, list of dict, list of dict, list of dict,
                  list of dict, list of dict, list of dict]:
                char {1,2,3}-gram, chartype {1,2,3}-gram and word feature-value
                dicts.
        """
        return (
            extract_char_from_tokens(tokens, 1, 3),
            extract_char_from_tokens(tokens, 2, 3),
            extract_char_from_tokens(tokens, 3, 3),
            extract_chartype_from_tokens(tokens, 1, 3),
            extract_chartype_from_tokens(tokens, 2, 3),
            extract_chartype_from_tokens(tokens, 3, 3),
            extract_word_features(tokens),
        )

    @staticmethod
    def extract_features(
        tokens,
        char1_ttoi,
        char2_ttoi,
        char3_ttoi,
        chartype1_ttoi,
        chartype2_ttoi,
        chartype3_ttoi,
        word_ttoi,
    ):
        """Extract indices of features.

        Args:
            tokens (list of str): Tokens.
        Returns:
            tuple[np.ndarray, shape=(len(tokens), 6),
                  np.ndarray, shape=(len(tokens), 5),
                  np.ndarray, shape=(len(tokens), 4),
                  np.ndarray, shape=(len(tokens), 6),
                  np.ndarray, shape=(len(tokens), 5),
                  np.ndarray, shape=(len(tokens), 4),
                  np.ndarray, shape=(len(tokens), 1)]:
                Indices of char {1,2,3}-gram, chartype {1,2,3}-gram and word
                features.
        """
        (
            char1_feature_dicts,
            char2_feature_dicts,
            char3_feature_dicts,
            type1_feature_dicts,
            type2_feature_dicts,
            type3_feature_dicts,
            word_feature_dicts,
        ) = RNNTagger.extract_feature_dicts(tokens)
        return (
            feature_dicts_to_indices(char1_feature_dicts, char1_ttoi),
            feature_dicts_to_indices(char2_feature_dicts, char2_ttoi),
            feature_dicts_to_indices(char3_feature_dicts, char3_ttoi),
            feature_dicts_to_indices(type1_feature_dicts, chartype1_ttoi),
            feature_dicts_to_indices(type2_feature_dicts, chartype2_ttoi),
            feature_dicts_to_indices(type3_feature_dicts, chartype3_ttoi),
            feature_dicts_to_indices(word_feature_dicts, word_ttoi),
        )


class LRRNNTagger:
    """A linear interpolation of LRTagger and RNNTagger.

    Args:
        lr_tagger (LRTagger): LRTagger instance.
        rnn_tagger (RNNTagger): RNNTagger instance.
        alpha (float): Alpha value for linear interpolation.
    """

    def __init__(self, lr_tagger, rnn_tagger, alpha):
        self._lr_tagger = lr_tagger
        self._rnn_tagger = rnn_tagger
        self._alpha = alpha

    def predict(self, tokens):
        """Predict tags.

        Args:
            tokens (list of str): Tokens.
        Returns:
            list of str: Predicted tags.
        """
        tags = []
        for score, itot, _ in self.predict_scores(tokens):
            if score.shape == (1, 1):
                tags.append(itot[0])
            else:
                tags.append(itot[np.argmax(score)])
        return tags

    def predict_scores(self, tokens):
        """Predict scores.

        Args:
            tokens (list of str): Tokens.
        Returns:
            list of tuple (np.ndarray, dict, dict): Scores, itots, and ttois.
        """
        scores = []
        rnn_probs = self._rnn_tagger.predict_log_proba(tokens)
        for i, _ in enumerate(tokens):
            lr_prob, lr_itot, lr_ttoi = self._lr_tagger.predict_single_log_proba(
                tokens, i
            )
            if lr_prob.shape == (1, 1):
                scores.append((lr_prob, lr_itot, lr_ttoi))
            else:
                rnn_ttoi = {t: self._rnn_tagger.ttoi[t] for t in lr_itot}
                rnn_prob = softmax([rnn_probs[i][id_] for _, id_ in rnn_ttoi.items()])
                lerp = lr_prob.reshape(-1) * (1 - self._alpha) + rnn_prob * self._alpha
                scores.append((lerp, lr_itot, lr_ttoi))
        return scores
