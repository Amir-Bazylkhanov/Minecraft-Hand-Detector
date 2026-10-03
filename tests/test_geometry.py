import math
import unittest

from handcraft.geometry import (HandClass, THUMB_GAP_EXTENDED,
                                THUMB_GAP_FOLDED, classify_hand, palm_size,
                                thumb_gap_ratio, thumb_hysteresis)
from tests.hands import make_hand


def transformed(hand, angle=0.0, mirror=False, scale=1.0, dx=0.0, dy=0.0):
    """Rotate (optionally mirror) a hand in the image plane, then scale
    and translate it — the variations a real camera feed introduces."""
    c, s = math.cos(angle), math.sin(angle)
    out = []
    for x, y, z in hand:
        if mirror:
            x = -x
        out.append(((x * c - y * s) * scale + dx,
                    (x * s + y * c) * scale + dy, z * scale))
    return out


class TestGeometry(unittest.TestCase):
    def test_open_palm_classified(self):
        self.assertIs(classify_hand(make_hand("open")), HandClass.OPEN_PALM)

    def test_fist_classified(self):
        self.assertIs(classify_hand(make_hand("fist")), HandClass.FIST)

    def test_scale_invariant(self):
        for scale in (0.3, 1.0, 3.0):
            self.assertIs(classify_hand(make_hand("open", scale=scale)),
                          HandClass.OPEN_PALM)
            self.assertIs(classify_hand(make_hand("fist", scale=scale)),
                          HandClass.FIST)

    def test_palm_size(self):
        self.assertAlmostEqual(palm_size(make_hand()), 0.2, places=5)
        self.assertAlmostEqual(palm_size(make_hand(scale=2.0)), 0.4, places=5)

    def test_degenerate_input_is_other(self):
        self.assertIs(classify_hand([(0.0, 0.0, 0.0)] * 21), HandClass.OTHER)
        self.assertIs(classify_hand([]), HandClass.OTHER)


class TestThumbGap(unittest.TestCase):
    def test_gap_ratio_extended_and_folded(self):
        extended = thumb_gap_ratio(make_hand("open"))
        folded = thumb_gap_ratio(make_hand("fist"))
        self.assertGreater(extended, THUMB_GAP_EXTENDED)
        self.assertLess(folded, THUMB_GAP_FOLDED)

    def test_gap_ratio_moderate_extension_is_extended(self):
        # thumb tip pulled to a .7 palm-normalized gap: under the legacy
        # .8 line, over the modern .65 one
        hand = make_hand("fist")
        mcp, palm = hand[5], palm_size(hand)
        vx, vy = hand[4][0] - mcp[0], hand[4][1] - mcp[1]
        length = math.hypot(vx, vy)
        hand[4] = (mcp[0] + vx / length * palm * .7,
                   mcp[1] + vy / length * palm * .7, mcp[2])
        self.assertGreater(thumb_gap_ratio(hand), THUMB_GAP_EXTENDED)

    def test_gap_ratio_degenerate_or_nonfinite_is_nan(self):
        self.assertTrue(math.isnan(thumb_gap_ratio([])))
        self.assertTrue(math.isnan(thumb_gap_ratio([(0.0, 0.0, 0.0)] * 21)))
        self.assertTrue(math.isnan(thumb_gap_ratio(
            [(float("nan"), 0.0, 0.0)] * 21)))
        hand = make_hand("open")
        self.assertTrue(math.isnan(thumb_gap_ratio(hand, palm=0.0)))
        self.assertTrue(math.isnan(thumb_gap_ratio(hand, palm=float("nan"))))

    def test_gap_ratio_invariant_to_mirror_rotation_scale_translation(self):
        for kind in ("open", "fist"):
            base = thumb_gap_ratio(make_hand(kind))
            for kwargs in (dict(dx=.4, dy=-.3),
                           dict(scale=2.5),
                           dict(mirror=True),
                           dict(angle=.9, mirror=True, scale=1.7,
                                dx=.2, dy=.1)):
                hand = transformed(make_hand(kind), **kwargs)
                self.assertAlmostEqual(thumb_gap_ratio(hand), base, places=7)

    def test_hysteresis_edges_and_band(self):
        out, rose, fell = thumb_hysteresis(.7, False)
        self.assertEqual((out, rose, fell), (True, True, False))
        # ambiguous .60 inside the band keeps either state, no edges
        self.assertEqual(thumb_hysteresis(.6, True), (True, False, False))
        self.assertEqual(thumb_hysteresis(.6, False), (False, False, False))
        out, rose, fell = thumb_hysteresis(.5, True)
        self.assertEqual((out, rose, fell), (False, False, True))
        # a .5 fold from rest is no edge
        self.assertEqual(thumb_hysteresis(.5, False), (False, False, False))

    def test_hysteresis_holds_state_on_nan(self):
        self.assertEqual(thumb_hysteresis(float("nan"), True), (True, False, False))
        self.assertEqual(thumb_hysteresis(float("nan"), False), (False, False, False))


if __name__ == "__main__":
    unittest.main()
