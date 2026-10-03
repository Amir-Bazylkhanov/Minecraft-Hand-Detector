"""Tests for handedness mapping and post-detection x mirroring."""

from types import SimpleNamespace
import unittest

from handcraft.tracker import CameraWorker


def _lm(x, y, z):
    return SimpleNamespace(x=x, y=y, z=z)


def _category(name, score):
    return SimpleNamespace(category_name=name, score=score)


def _result(pairs):
    return SimpleNamespace(
        hand_landmarks=[p[0] for p in pairs],
        handedness=[[p[1]] for p in pairs],
    )


class HandsByHandednessTest(unittest.TestCase):
    def test_labels_preserved_x_mirrored_yz_unchanged(self):
        left_landmarks = [_lm(0.2, 0.5, -0.01) for _ in range(21)]
        right_landmarks = [_lm(0.8, 0.3, 0.02) for _ in range(21)]
        result = _result([
            (left_landmarks, _category("Left", 0.9)),
            (right_landmarks, _category("Right", 0.8)),
        ])

        hands = CameraWorker._hands_by_handedness(result)

        self.assertEqual(set(hands), {"left", "right"})
        for x, y, z in hands["left"]:
            self.assertAlmostEqual(x, 0.8)      # 1.0 - 0.2
            self.assertAlmostEqual(y, 0.5)
            self.assertAlmostEqual(z, -0.01)
        for x, y, z in hands["right"]:
            self.assertAlmostEqual(x, 0.2)      # 1.0 - 0.8
            self.assertAlmostEqual(y, 0.3)
            self.assertAlmostEqual(z, 0.02)

    def test_higher_confidence_duplicate_wins(self):
        low = [_lm(0.1, 0.1, 0.0) for _ in range(21)]
        high = [_lm(0.4, 0.6, 0.0) for _ in range(21)]
        result = _result([
            (low, _category("Left", 0.5)),
            (high, _category("Left", 0.9)),
        ])

        hands = CameraWorker._hands_by_handedness(result)

        self.assertEqual(list(hands), ["left"])
        x, y, _ = hands["left"][0]
        self.assertAlmostEqual(x, 0.6)  # 1.0 - 0.4 from the high-score entry
        self.assertAlmostEqual(y, 0.6)

    def test_lower_confidence_duplicate_loses(self):
        high = [_lm(0.4, 0.6, 0.0) for _ in range(21)]
        low = [_lm(0.1, 0.1, 0.0) for _ in range(21)]
        result = _result([
            (high, _category("Right", 0.9)),
            (low, _category("Right", 0.5)),
        ])

        hands = CameraWorker._hands_by_handedness(result)

        x, y, _ = hands["right"][0]
        self.assertAlmostEqual(x, 0.6)
        self.assertAlmostEqual(y, 0.6)


if __name__ == "__main__":
    unittest.main()
