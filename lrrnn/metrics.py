from ignite.metrics import Metric

from . import info


def read_corpus(corpus):
    """Read a prediction or groundtruth file.

    Args:
        corpus (str): Path to the prediction or groundtruth file.
    Returns:
        list of list of list of str: List of sentences.
            Each sentence is a list of morphs.
            Each morph is a list of morpheme information.
    """
    sentences = []
    with open(corpus, encoding=info.charset) as f:
        for line in f:
            morphs = line.replace(info.newline, "").split()
            sentences.append([m.split(info.sep) for m in morphs])
    return sentences


def get_min_max_level(sentences):
    """Get the minimum and maximum number of morpheme information.

    Args:
        sentences (list of list of list of str): List of sentences.
            Each sentence is a list of morphs.
            Each morph is a list of morpheme information.
    Returns:
        tuple[int, int]: Minimum and maximum number of morpheme information.
    """
    max_number_of_morpheme_information = float("-inf")
    min_number_of_morpheme_information = float("inf")

    for sentence in sentences:
        for morph in sentence:
            max_number_of_morpheme_information = max(
                max_number_of_morpheme_information, len(morph)
            )
            min_number_of_morpheme_information = min(
                min_number_of_morpheme_information, len(morph)
            )

    min_level = min_number_of_morpheme_information - 1
    max_level = max_number_of_morpheme_information - 1
    return (min_level, max_level)


def validate(predictions, groundtruths):
    """Validate predictions and groundtruths.

    This function checks:
    - Number of sentences between predictions and groundtruths are identical.
    - Number of morpheme information in predictions are consistent.
    - Number of morpheme information in groundtruths are consistent.
    - Number of morpheme information between predictions and groundtruths are consistent.

    Args:
        predictions (list of list of list of str): List of predicted sentences.
            Each sentence is a list of morphs.
            Each morph is a list of morpheme information.
        groundtruths (list of list of list of str): List of groundtruth sentences.
            Each sentence is a list of morphs.
            Each morph is a list of morpheme information.
    """
    if len(predictions) != len(groundtruths):
        raise ValueError(
            "Number of sentences between predictions and groundtruths are different"
        )

    min_prediction_level, max_prediction_level = get_min_max_level(predictions)
    if min_prediction_level != max_prediction_level:
        raise ValueError(
            "Number of morpheme information in predictions are not consistent"
        )

    min_groundtruth_level, max_groundtruth_level = get_min_max_level(groundtruths)
    if min_groundtruth_level != max_groundtruth_level:
        raise ValueError(
            "Number of morpheme information in groundtruths are not consistent"
        )

    if min_prediction_level != min_groundtruth_level:
        raise ValueError(
            "Number of morpheme information between predictions and groundtruths are different"
        )


def clip_morpheme_information(sentence, level):
    """Clip morpheme information to the specified level.

    Args:
        sentence (list of list of str): List of morphs.
            Each morph is a list of morpheme information.
        level (int): Level to clip morpheme information.
    Returns:
        list of list of str: Clipped morphs.
    """
    if level < 0:
        raise ValueError("Level must be greater than or equal to 0")
    return [morph[: level + 1] for morph in sentence]


def add_unique_identifiers(sentence):
    """Add unique identifiers to morphs for evaluation.

    Args:
        sentence (list of list of str): List of morphs.
            Each morph is a list of morpheme information.
    Returns:
        list of tuple[int, str]: Uniquely identifiable morphs.
    """
    eval_morphs = []

    num_characters_sum = 0
    for morph in sentence:
        eval_morphs.append((num_characters_sum, info.sep.join(morph)))
        num_characters_sum += len(morph[0])

    return eval_morphs


def compute_precision_recall_f(predictions, groundtruths, level):
    """Compute precision, recall, and f1-score at the specified level.

    Args:
        predictions (list of list of list of str): List of predicted sentences.
            Each sentence is a list of morphs.
            Each morph is a list of morpheme information.
        groundtruths (list of list of list of str): List of groundtruth sentences.
            Each sentence is a list of morphs.
            Each morph is a list of morpheme information.
        level (int): Level to compute precision, recall, and f1-score.
    Returns:
        tuple[float, float, float]: Precision, Recall, and F1-score.
    """
    num_predicted_morphs = 0
    num_groundtruth_morphs = 0
    num_correct_morphs = 0

    for predicted_sentence, groundtruth_sentence in zip(predictions, groundtruths):
        predicted_morphs = add_unique_identifiers(
            clip_morpheme_information(predicted_sentence, level)
        )
        groundtruth_morphs = add_unique_identifiers(
            clip_morpheme_information(groundtruth_sentence, level)
        )
        num_predicted_morphs += len(predicted_morphs)
        num_groundtruth_morphs += len(groundtruth_morphs)
        num_correct_morphs += sum(
            [1 for predicted_m in predicted_morphs if predicted_m in groundtruth_morphs]
        )

    precision = num_correct_morphs / num_predicted_morphs
    recall = num_correct_morphs / num_groundtruth_morphs

    # Avoid division by zero
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = (2 * precision * recall) / (precision + recall)

    return (precision, recall, f1_score)


def evaluate(predictions, groundtruths):
    """Shortcut function to compute precision, recall and F1-score at each level.

    Args:
        predictions (list of list of list of str): List of predicted sentences.
        groundtruths (list of list of list of str): List of groundtruth sentences.
    Returns:
        list of tuple[float, float, float]: List of precision, recall, and f1-score at each level.
    """
    scores = []
    validate(predictions, groundtruths)
    _, max_level = get_min_max_level(predictions)
    for level in range(max_level + 1):
        scores.append(compute_precision_recall_f(predictions, groundtruths, level))
    return scores


class F1(Metric):
    def __init__(self):
        self._predictions = []
        self._groundtruths = []
        super().__init__()

    def reset(self):
        self._predictions = []
        self._groundtruths = []
        super().reset()

    def update(self, output):
        self._predictions.append(output[0])
        self._groundtruths.append(output[1])

    def compute(self):
        scores = evaluate(self._predictions, self._groundtruths)
        _, _, f1 = scores[-1]
        return f1
