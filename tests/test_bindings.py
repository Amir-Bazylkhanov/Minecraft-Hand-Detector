import unittest

from handcraft.bindings import PalmContactRecognizer


class AttrPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def synthetic_hand(index_distance=1.4, thumb_distance=1.4, scale=1.0, offset=(0.0, 0.0), as_objects=False):
    base = [(1.2, -1.2) for _ in range(21)]
    base[0] = (0.0, 0.0)
    base[5] = (-0.45, -0.75)
    base[9] = (0.0, -1.0)
    base[13] = (0.4, -0.85)
    base[17] = (0.7, -0.7)

    palm_points = [base[index] for index in (0, 5, 9, 13, 17)]
    palm_x = sum(point[0] for point in palm_points) / 5
    palm_y = sum(point[1] for point in palm_points) / 5
    palm_scale = (sum((x - palm_x) ** 2 + (y - palm_y) ** 2 for x, y in palm_points) / 5) ** 0.5

    base[8] = (palm_x, palm_y - index_distance * palm_scale)
    base[4] = (palm_x - thumb_distance * palm_scale, palm_y)
    transformed = [
        (x * scale + offset[0], y * scale + offset[1])
        for x, y in base
    ]
    if as_objects:
        return [AttrPoint(x, y) for x, y in transformed]
    return transformed


class PalmContactRecognizerTests(unittest.TestCase):
    def test_scale_invariance_and_attribute_points(self):
        for scale, offset in ((0.35, (-20.0, 40.0)), (1.0, (0.0, 0.0)), (3.0, (500.0, -250.0))):
            with self.subTest(scale=scale):
                recognizer = PalmContactRecognizer()
                hand = synthetic_hand(
                    index_distance=0.10,
                    thumb_distance=1.3,
                    scale=scale,
                    offset=offset,
                    as_objects=True,
                )
                result = recognizer.update_hands({"Left": hand})
                self.assertEqual(result.held_keys, frozenset({"W"}))
                self.assertAlmostEqual(result.distances["left_index_palm"], 0.10, places=6)

    def test_enter_exit_hysteresis(self):
        recognizer = PalmContactRecognizer()

        result = recognizer.update_hands({"Left": synthetic_hand(index_distance=0.70)})
        self.assertNotIn("W", result.held_keys)

        result = recognizer.update_hands({"Left": synthetic_hand(index_distance=0.10)})
        self.assertIn("W", result.held_keys)

        result = recognizer.update_hands({"Left": synthetic_hand(index_distance=0.40)})
        self.assertIn("W", result.held_keys, "between thresholds must preserve contact")

        result = recognizer.update_hands({"Left": synthetic_hand(index_distance=0.55)})
        self.assertNotIn("W", result.held_keys)

        result = recognizer.update_hands({"Left": synthetic_hand(index_distance=0.40)})
        self.assertNotIn("W", result.held_keys, "between thresholds must preserve release")

    def test_tracking_loss_releases_hold_and_reacquisition_recovers(self):
        recognizer = PalmContactRecognizer()
        contact = {"Left": synthetic_hand(index_distance=0.10, thumb_distance=1.3)}

        self.assertEqual(recognizer.update_hands(contact).held_keys, frozenset({"W"}))
        lost = recognizer.update_hands({})
        self.assertEqual(lost.held_keys, frozenset())
        self.assertEqual(lost.tracked_hands, frozenset())
        self.assertFalse(lost.contact_states["left_index_palm"])
        self.assertIsNone(lost.distances["left_index_palm"])

        recovered = recognizer.update_hands(contact)
        self.assertEqual(recovered.held_keys, frozenset({"W"}))

    def test_thumb_pulse_is_one_shot_per_contact(self):
        recognizer = PalmContactRecognizer()
        thumb_contact = {"Left": synthetic_hand(index_distance=1.3, thumb_distance=0.10)}

        first = recognizer.update_hands(thumb_contact)
        self.assertEqual(first.pulse_keys, ("SPACE",))
        self.assertEqual(first.held_keys, frozenset())

        repeated = recognizer.update_hands(thumb_contact)
        self.assertEqual(repeated.pulse_keys, ())
        self.assertTrue(repeated.contact_states["left_thumb_palm"])

        released = recognizer.update_hands({"Left": synthetic_hand(index_distance=1.3, thumb_distance=0.60)})
        self.assertEqual(released.pulse_keys, ())
        self.assertFalse(released.contact_states["left_thumb_palm"])

        second = recognizer.update_hands(thumb_contact)
        self.assertEqual(second.pulse_keys, ("SPACE",))

    def test_handedness_is_separate_and_mappings_are_configurable(self):
        recognizer = PalmContactRecognizer()
        right_contact = {"Right": synthetic_hand(index_distance=0.10, thumb_distance=0.10)}
        self.assertEqual(recognizer.update_hands(right_contact).held_keys, frozenset())
        self.assertEqual(recognizer.update_hands(right_contact).pulse_keys, ())

        custom = PalmContactRecognizer({
            "right_index_palm": {
                "hand": "right",
                "finger": "index",
                "action": "hold",
                "key": "E",
            }
        })
        left_contact = {"Left": synthetic_hand(index_distance=0.10, thumb_distance=1.3)}
        self.assertEqual(custom.update_hands(left_contact).held_keys, frozenset())
        self.assertEqual(
            custom.update_hands({"Right": synthetic_hand(index_distance=0.10, thumb_distance=1.3)}).held_keys,
            frozenset({"E"}),
        )

    def test_index_thumb_ambiguity_does_not_enter_contact(self):
        recognizer = PalmContactRecognizer()
        both_tips_touching = {"Left": synthetic_hand(index_distance=0.10, thumb_distance=0.10)}
        result = recognizer.update_hands(both_tips_touching)
        self.assertEqual(result.held_keys, frozenset())
        self.assertEqual(result.pulse_keys, ())
        self.assertFalse(result.contact_states["left_index_palm"])
        self.assertFalse(result.contact_states["left_thumb_palm"])

    def test_confidence_is_bounded(self):
        recognizer = PalmContactRecognizer()
        for distance in (0.0, 0.10, 0.32, 0.40, 0.48, 0.80):
            result = recognizer.update_hands({"Left": synthetic_hand(index_distance=distance, thumb_distance=1.3)})
            for confidence in result.confidences.values():
                self.assertGreaterEqual(confidence, 0.0)
                self.assertLessEqual(confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
