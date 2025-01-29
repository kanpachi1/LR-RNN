import math
import tempfile
import unittest

from lrrnn import metrics


class TestMetricsMethods(unittest.TestCase):

    def setUp(self):
        self.groundtruths = [
            [
                ["Good", "adjective"],
                ["things", "noun"],
                ["come", "verb"],
                ["in", "preposition"],
                ["small", "adjective"],
                ["packages", "noun"],
                [".", "period"],
            ]
        ]
        self.predictions = [
            [
                ["Good", "adjective"],
                ["things", "noun"],
                ["come", "verb"],
                ["in", "preposition"],
                ["small", "adverb"],
                ["pack", "noun"],
                ["ages", "noun"],
                [".", "period"],
            ]
        ]

    def test_read_corpus(self):
        corpus = tempfile.mkstemp()
        with open(corpus[1], "w") as f:
            f.write(
                "Good/adjective things/noun come/verb in/preposition "
                "small/adjective packages/noun ./period\n"
            )
        self.assertEqual(metrics.read_corpus(corpus[1]), self.groundtruths)

    def test_get_min_max_level(self):
        self.assertEqual(metrics.get_min_max_level(self.groundtruths), (1, 1))
        self.assertEqual(metrics.get_min_max_level(self.predictions), (1, 1))

        self.groundtruths[0][0].append("extra")
        self.assertEqual(metrics.get_min_max_level(self.groundtruths), (1, 2))
        self.groundtruths[0][0].pop()  # Reset

        self.groundtruths[0][0].pop()
        self.assertEqual(metrics.get_min_max_level(self.groundtruths), (0, 1))

    def test_validate(self):
        try:
            metrics.validate(self.predictions, self.groundtruths)
        except ValueError:
            self.fail("validate() raised ValueError unexpectedly!")

        with self.assertRaises(ValueError) as cm:
            metrics.validate(self.predictions, [])
        self.assertEqual(
            cm.exception.args[0],
            "Number of sentences between predictions and groundtruths are different",
        )

        self.predictions[0][0].append("extra")
        with self.assertRaises(ValueError) as cm:
            metrics.validate(self.predictions, self.groundtruths)
        self.assertEqual(
            cm.exception.args[0],
            "Number of morpheme information in predictions are not consistent",
        )
        self.predictions[0][0].pop()  # Reset

        self.groundtruths[0][0].append("extra")
        with self.assertRaises(ValueError) as cm:
            metrics.validate(self.predictions, self.groundtruths)
        self.assertEqual(
            cm.exception.args[0],
            "Number of morpheme information in groundtruths are not consistent",
        )
        self.groundtruths[0][0].pop()  # Reset

        with self.assertRaises(ValueError) as cm:
            metrics.validate(self.predictions, [[["extra", "info", "exists"]]])
        self.assertEqual(
            cm.exception.args[0],
            "Number of morpheme information between predictions and groundtruths are different",
        )

    def test_clip_morpheme_information(self):
        with self.assertRaises(ValueError) as cm:
            metrics.clip_morpheme_information(self.groundtruths[0], -1)
        self.assertEqual(
            cm.exception.args[0],
            "Level must be greater than or equal to 0",
        )

        self.assertEqual(
            metrics.clip_morpheme_information(self.groundtruths[0], 1),
            [
                ["Good", "adjective"],
                ["things", "noun"],
                ["come", "verb"],
                ["in", "preposition"],
                ["small", "adjective"],
                ["packages", "noun"],
                [".", "period"],
            ],
        )
        self.assertEqual(
            metrics.clip_morpheme_information(self.groundtruths[0], 0),
            [
                ["Good"],
                ["things"],
                ["come"],
                ["in"],
                ["small"],
                ["packages"],
                ["."],
            ],
        )

    def test_add_unique_identifiers(self):
        self.assertEqual(
            metrics.add_unique_identifiers(self.groundtruths[0]),
            [
                (0, "Good/adjective"),
                (4, "things/noun"),
                (10, "come/verb"),
                (14, "in/preposition"),
                (16, "small/adjective"),
                (21, "packages/noun"),
                (29, "./period"),
            ],
        )

    def test_compute_precision_recall_f(self):
        # level=1
        p, r, f = metrics.compute_precision_recall_f(
            self.predictions, self.groundtruths, 1
        )
        n_correct_morphs = 5
        precision = n_correct_morphs / len(self.predictions[0])
        recall = n_correct_morphs / len(self.groundtruths[0])
        f1_score = (2 * precision * recall) / (precision + recall)
        self.assertTrue(math.isclose(p, precision))
        self.assertTrue(math.isclose(r, recall))
        self.assertTrue(math.isclose(f, f1_score))

        # level=0
        p, r, f = metrics.compute_precision_recall_f(
            self.predictions, self.groundtruths, 0
        )
        n_correct_morphs = 6
        precision = n_correct_morphs / len(self.predictions[0])
        recall = n_correct_morphs / len(self.groundtruths[0])
        f1_score = (2 * precision * recall) / (precision + recall)
        self.assertTrue(math.isclose(p, precision))
        self.assertTrue(math.isclose(r, recall))
        self.assertTrue(math.isclose(f, f1_score))

        # Zero division
        self.assertEqual(
            metrics.compute_precision_recall_f(
                [[["morph", "pos"]]], self.groundtruths, 1
            ),
            (0, 0, 0),
        )

    def test_evaluate(self):
        print(self.predictions)
        print(self.groundtruths)
        scores = metrics.evaluate(self.predictions, self.groundtruths)
        print(scores)
        self.assertEqual(len(scores), 2)
