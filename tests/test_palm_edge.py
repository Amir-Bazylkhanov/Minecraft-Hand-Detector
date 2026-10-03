"""Regression for the user's folded index at the outside palm edge."""
import unittest

from handcraft.bindings import PalmContactRecognizer


def pictured_hand():
    # Approximate screenshot positions, including visibly extended fingers.
    points = [(0.0, 0.0, 0.0)] * 21
    positions = {0: (300, 361), 5: (358, 196), 6: (353, 234),
                 7: (349, 264), 8: (346, 282), 9: (298, 204),
                 10: (312, 129), 12: (329, 49), 13: (256, 211),
                 14: (243, 123), 16: (238, 40), 17: (224, 232),
                 18: (194, 159), 20: (175, 96), 4: (432, 186)}
    for index, (x, y) in positions.items():
        points[index] = (x / 748, y / 512, 0.0)
    return points


class PalmEdgeTests(unittest.TestCase):
    def test_pictured_folded_index_holds_walk_and_no_jump(self):
        result = PalmContactRecognizer().update_hands({'left': pictured_hand()})
        self.assertEqual(result.held_keys, frozenset({'W'}))
        self.assertEqual(result.pulse_keys, ())
        self.assertGreater(result.confidences['left_index_palm'], 0.5)

    def test_right_hand_does_not_walk(self):
        result = PalmContactRecognizer().update_hands({'right': pictured_hand()})
        self.assertFalse(result.held_keys)

    def test_straight_index_near_edge_does_not_walk(self):
        points = pictured_hand()
        points[6] = (320 / 748, 315 / 512, 0)
        result = PalmContactRecognizer().update_hands({'left': points})
        self.assertFalse(result.held_keys)

    def test_full_fist_releases_edge_hold(self):
        recognizer = PalmContactRecognizer()
        points = pictured_hand()
        self.assertEqual(recognizer.update_hands({'left': points}).held_keys,
                         frozenset({'W'}))
        for tip in (12, 16, 20):
            points[tip] = (300 / 748, 285 / 512, 0)
        self.assertFalse(recognizer.update_hands({'left': points}).held_keys)

    def test_far_folded_index_does_not_walk(self):
        points = pictured_hand()
        points[8] = (397 / 748, 282 / 512, 0)
        self.assertFalse(PalmContactRecognizer().update_hands({'left': points}).held_keys)

    def test_edge_hysteresis_and_tracking_loss_release(self):
        recognizer = PalmContactRecognizer()
        points = pictured_hand()
        self.assertTrue(recognizer.update_hands({'left': points}).held_keys)
        points[8] = (358 / 748, 282 / 512, 0)
        self.assertFalse(PalmContactRecognizer().update_hands({'left': points}).held_keys)
        self.assertTrue(recognizer.update_hands({'left': points}).held_keys)
        points[8] = (366 / 748, 282 / 512, 0)
        self.assertFalse(recognizer.update_hands({'left': points}).held_keys)
        self.assertTrue(recognizer.update_hands({'left': pictured_hand()}).held_keys)
        self.assertFalse(recognizer.update_hands({}).held_keys)

    def test_geometry_is_scale_translation_and_reflection_invariant(self):
        for scale in (0.2, 1.0, 3.0, -1.0):
            points = [(x * scale + 0.7, y * abs(scale) - 0.2, z)
                      for x, y, z in pictured_hand()]
            result = PalmContactRecognizer().update_hands({'left': points})
            self.assertEqual(result.held_keys, frozenset({'W'}))


if __name__ == '__main__':
    unittest.main()
