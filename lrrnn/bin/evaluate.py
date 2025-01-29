import argparse

from lrrnn import metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute precision, recall and f1-score for a given prediction"
    )
    parser.add_argument(
        "prediction",
        help="path to the prediction file (in the same format as the KyTea output)",
    )
    parser.add_argument(
        "groundtruth",
        help="path to the groundtruth file (in the same format as the KyTea output)",
    )
    args = parser.parse_args()

    predictions = metrics.read_corpus(args.prediction)
    groundtruths = metrics.read_corpus(args.groundtruth)
    scores = metrics.evaluate(predictions, groundtruths)

    print("level\tprec\trecall\tf1-score")
    for level, (precision, recall, f) in enumerate(scores):
        print(
            "{}\t{:.2f}\t{:.2f}\t{:.2f}".format(
                level, precision * 100, recall * 100, f * 100
            )
        )
