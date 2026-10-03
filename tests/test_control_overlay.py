"""Tests for handcraft.control_overlay with cv2 fully mocked.

Stdlib unittest only, so it runs under ``.venv python -m unittest``
without pytest or numpy (cv2 is mocked, so a stub image suffices).
"""
import queue
import re
import sys
import unittest
from unittest import mock

from handcraft import control_overlay
from handcraft.control_overlay import draw_control_overlay
from handcraft.tracker import CameraWorker

W, H = 640, 480
FONT_THICKNESS = control_overlay._THICKNESS


class FakeImage:
    """Stands in for an RGB ndarray; only .shape is used (cv2 is mocked)."""
    shape = (H, W, 3)


def make_hand(x=0.3, y=0.5, thumb=None):
    """21 plausible landmarks around wrist (x, y) with a real palm span."""
    pts = [(x, y, 0.0)] * 21
    pts[5] = (x + 0.05, y - 0.08, 0.0)
    pts[9] = (x + 0.02, y - 0.10, 0.0)
    pts[13] = (x - 0.01, y - 0.09, 0.0)
    pts[17] = (x - 0.05, y - 0.08, 0.0)
    if thumb is not None:
        pts[4] = thumb
    return pts


def fake_text_size(text, font, scale, thickness):
    """Deterministic getTextSize stand-in: size grows with the text."""
    return ((max(1, len(text)) * 9, 12), 4)


def base_state(**over):
    state = {
        "left_center": (0.3, 0.5),
        "right_center": (0.7, 0.5),
        "held_keys": (),
        "labels": {},
        "inventory_open": False,
        "hotbar_active": False,
        "left_ready": True,
    }
    state.update(over)
    return state


class ControlOverlayTest(unittest.TestCase):
    def setUp(self):
        self.cv2 = mock.MagicMock()
        self.cv2.getTextSize.side_effect = fake_text_size
        patcher = mock.patch.dict(sys.modules, {"cv2": self.cv2})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.img = FakeImage()

    def texts(self):
        return [c.args[1] for c in self.cv2.putText.call_args_list]

    def by_text(self):
        return {c.args[1]: c for c in self.cv2.putText.call_args_list}

    def test_direction_captions_drawn(self):
        hands = {"left": make_hand(0.3, 0.5)}
        draw_control_overlay(self.img, hands, base_state())
        drawn = self.texts()
        for caption in ("W Forward", "S Back", "A Left", "D Right"):
            self.assertIn(caption, drawn)

    def test_active_key_bright_green(self):
        hands = {"left": make_hand(0.3, 0.5)}
        draw_control_overlay(self.img, hands, base_state(held_keys=("W",)))
        calls = self.by_text()
        self.assertEqual(calls["W Forward"].args[5],
                         control_overlay.ACTIVE_COLOR)
        for caption in ("S Back", "A Left", "D Right"):
            self.assertEqual(calls[caption].args[5],
                             control_overlay.NEUTRAL_COLOR)

    def test_wrist_marker_coordinates(self):
        hands = {"left": make_hand(0.3, 0.5)}
        draw_control_overlay(self.img, hands, base_state())
        centers = [c.args[1] for c in self.cv2.circle.call_args_list]
        self.assertIn((int(0.3 * W), int(0.5 * H)), centers)

    def test_left_zone_rectangle_at_center(self):
        from handcraft.geometry import palm_size
        hand = make_hand(0.25, 0.4)
        palm = palm_size(hand)
        draw_control_overlay(self.img, {"left": hand},
                             base_state(left_center=(0.25, 0.4)))
        cx, cy = 0.25 * W, 0.4 * H
        ex = control_overlay.ENTER_DEADZONE * palm * W
        ey = control_overlay.ENTER_DEADZONE * palm * H
        expected = ((int(cx - ex), int(cy - ey)), (int(cx + ex), int(cy + ey)))
        rects = [(c.args[1], c.args[2])
                 for c in self.cv2.rectangle.call_args_list
                 if c.args[3] == control_overlay.LEFT_ZONE_COLOR]
        self.assertIn(expected, rects)

    def test_inventory_hides_zones_with_caption(self):
        hands = {"left": make_hand(0.3, 0.5), "right": make_hand(0.7, 0.5)}
        draw_control_overlay(self.img, hands,
                             base_state(inventory_open=True))
        zone_rects = [c for c in self.cv2.rectangle.call_args_list
                      if c.args[3] in (control_overlay.LEFT_ZONE_COLOR,
                                       control_overlay.RIGHT_ZONE_COLOR)]
        self.assertEqual(zone_rects, [])
        self.cv2.arrowedLine.assert_not_called()
        drawn = self.texts()
        self.assertTrue(
            any("Point to cursor" in t and "thumb out-in" in t
                and "V" not in t for t in drawn))
        self.assertTrue(any("inventory" in t for t in drawn))

    def test_attack_hint_gameplay_non_hotbar(self):
        """Gameplay without hotbar: right thumb cycles attack."""
        hands = {"right": make_hand(0.7, 0.5)}
        draw_control_overlay(self.img, hands, base_state())
        calls = self.by_text()
        self.assertIn("ATTACK: thumb out-in", calls)
        self.assertNotIn("CLICK: thumb out-in", calls)

    def test_click_hint_inventory(self):
        """Inventory: the same thumb cycle left-clicks instead."""
        hands = {"right": make_hand(0.7, 0.5)}
        draw_control_overlay(self.img, hands,
                             base_state(inventory_open=True))
        calls = self.by_text()
        self.assertIn("CLICK: thumb out-in", calls)
        self.assertNotIn("ATTACK: thumb out-in", calls)

    def test_thumb_hint_hidden_while_hotbar(self):
        """Hotbar mode owns the thumb: folded thumb steps slots, so no
        attack/click hint is shown in either mode."""
        for state in (base_state(hotbar_active=True),
                      base_state(hotbar_active=True, inventory_open=True)):
            self.cv2.putText.reset_mock()
            draw_control_overlay(self.img, {"right": make_hand(0.7, 0.5)},
                                 state)
            drawn = self.texts()
            self.assertNotIn("ATTACK: thumb out-in", drawn)
            self.assertNotIn("CLICK: thumb out-in", drawn)

    def test_hotbar_tip_labels(self):
        hands = {"right": make_hand(0.7, 0.5)}
        draw_control_overlay(self.img, hands, base_state(hotbar_active=True))
        drawn = self.texts()
        self.assertIn("Prev slot", drawn)
        self.assertIn("Next slot", drawn)

    def test_look_zone_caption(self):
        hands = {"right": make_hand(0.7, 0.5)}
        draw_control_overlay(self.img, hands, base_state())
        self.assertIn("LOOK", self.texts())
        look_rects = [c for c in self.cv2.rectangle.call_args_list
                      if c.args[3] == control_overlay.RIGHT_ZONE_COLOR]
        self.assertEqual(len(look_rects), 1)

    def test_left_not_ready_caption(self):
        hands = {"left": make_hand(0.3, 0.5)}
        draw_control_overlay(self.img, hands, base_state(left_ready=False))
        self.assertIn("Hold point still to center", self.texts())

    def test_mode_label_with_held_keys(self):
        draw_control_overlay(self.img, {"left": make_hand()},
                             base_state(held_keys=("W", "SHIFT")))
        drawn = self.texts()
        self.assertTrue(
            any("Mode: gameplay" in t and "W+SHIFT" in t for t in drawn))

    def test_finger_labels_drawn(self):
        hands = {"left": make_hand(0.3, 0.5), "right": make_hand(0.7, 0.5)}
        draw_control_overlay(self.img, hands,
                             base_state(labels={"left": "Move joystick",
                                                "right": "Look"}))
        drawn = self.texts()
        self.assertIn("Move joystick", drawn)
        self.assertIn("Look", drawn)

    def test_direction_caption_placement(self):
        """W/S centered on cx; A/D fully outside the box with an 8px gap."""
        from handcraft.geometry import palm_size
        hand = make_hand(0.3, 0.5)
        draw_control_overlay(self.img, {"left": hand}, base_state())
        cx = 0.3 * W
        ex = control_overlay.ENTER_DEADZONE * palm_size(hand) * W
        calls = self.by_text()

        def expected_x(caption, formula):
            tw = fake_text_size(caption, None, None, None)[0][0]
            return int(formula(tw))

        self.assertEqual(calls["W Forward"].args[2][0],
                         expected_x("W Forward", lambda tw: cx - tw / 2))
        self.assertEqual(calls["S Back"].args[2][0],
                         expected_x("S Back", lambda tw: cx - tw / 2))
        self.assertEqual(calls["A Left"].args[2][0],
                         expected_x("A Left", lambda tw: cx - ex - 8 - tw))
        self.assertEqual(calls["D Right"].args[2][0],
                         expected_x("D Right", lambda tw: cx + ex + 8))

    def test_jump_hint_neutral_thumb_out(self):
        """No SPACE held: the hint invites the thumb out, in neutral gray."""
        hands = {"left": make_hand(0.3, 0.5)}
        draw_control_overlay(self.img, hands, base_state())
        calls = self.by_text()
        self.assertIn("JUMP: thumb out", calls)
        self.assertEqual(calls["JUMP: thumb out"].args[5],
                         control_overlay.NEUTRAL_COLOR)
        self.assertNotIn("JUMP: SPACE held", calls)

    def test_jump_hint_space_held_green(self):
        """SPACE in held_keys: the hint reports the real held key, green."""
        hands = {"left": make_hand(0.3, 0.5)}
        draw_control_overlay(self.img, hands, base_state(held_keys=("SPACE",)))
        calls = self.by_text()
        self.assertIn("JUMP: SPACE held", calls)
        self.assertEqual(calls["JUMP: SPACE held"].args[5],
                         control_overlay.ACTIVE_COLOR)
        self.assertNotIn("JUMP: thumb out", calls)

    def test_jump_hint_left_of_thumb_tip_full_caption_width(self):
        """The full dynamic caption is measured, ending 8px left of the
        thumb tip (landmark 4), for both text variants."""
        hand = make_hand(0.3, 0.5)
        for state, caption in ((base_state(), "JUMP: thumb out"),
                               (base_state(held_keys=("SPACE",)),
                                "JUMP: SPACE held")):
            self.cv2.putText.reset_mock()
            draw_control_overlay(self.img, {"left": hand}, state)
            calls = self.by_text()
            tw = fake_text_size(caption, None, None, None)[0][0]
            self.assertEqual(calls[caption].args[2],
                             (int(hand[4][0] * W - tw - 8),
                              int(hand[4][1] * H)))

    def test_center_cross_and_displacement_line(self):
        """Fixed neutral anchor (cross) vs current wrist marker, linked."""
        hand = make_hand(0.35, 0.55)  # wrist offset from the center
        draw_control_overlay(self.img, {"left": hand},
                             base_state(left_center=(0.3, 0.5)))
        icx, icy = int(0.3 * W), int(0.5 * H)
        iwx, iwy = int(0.35 * W), int(0.55 * H)
        lines = [(c.args[1], c.args[2]) for c in self.cv2.line.call_args_list
                 if c.args[3] == control_overlay.LEFT_ZONE_COLOR]
        self.assertIn(((icx - 5, icy), (icx + 5, icy)), lines)  # cross h
        self.assertIn(((icx, icy - 5), (icx, icy + 5)), lines)  # cross v
        self.assertIn(((icx, icy), (iwx, iwy)), lines)          # offset
        centers = [c.args[1] for c in self.cv2.circle.call_args_list]
        self.assertIn((iwx, iwy), centers)  # wrist marker at the wrist

    def test_top_caption_and_label_rows_do_not_overlap(self):
        hands = {"left": make_hand(0.3, 0.5), "right": make_hand(0.7, 0.5)}
        draw_control_overlay(self.img, hands,
                             base_state(labels={"left": "Move joystick",
                                                "right": "Look"}))
        calls = self.by_text()
        self.assertIn("Move your WRIST; open hand to recenter", calls)
        caption_y = calls["Move your WRIST; open hand to recenter"].args[2][1]
        left_y = calls["Move joystick"].args[2][1]
        right_y = calls["Look"].args[2][1]
        # Distinct stacked top rows, all clear of the bottom zone area.
        self.assertEqual(len({caption_y, left_y, right_y}), 3)
        for y in (caption_y, left_y, right_y):
            self.assertLessEqual(y, 70)

    def test_no_hands_no_crash(self):
        draw_control_overlay(self.img, {}, base_state())
        draw_control_overlay(self.img, None, base_state())
        draw_control_overlay(self.img, None, None)
        draw_control_overlay(None, None, None)  # no frame: pure no-op

    def test_invalid_hand_skipped(self):
        bad = [(float("nan"), 0.5, 0.0)] * 21
        draw_control_overlay(self.img, {"left": bad},
                             base_state(left_ready=False))
        drawn = self.texts()
        self.assertNotIn("W Forward", drawn)
        self.assertNotIn("Hold point still to center", drawn)

    def test_text_stays_in_bounds_at_edges(self):
        """Regression: captions near frame corners must not overflow."""
        edge_state = base_state(left_center=(0.0, 0.0),
                                right_center=(1.0, 1.0),
                                hotbar_active=True,
                                labels={"left": "Move joystick",
                                        "right": "Look"})
        hands = {"left": make_hand(0.0, 0.0), "right": make_hand(1.0, 1.0)}
        draw_control_overlay(self.img, hands, edge_state)
        self.assertTrue(self.cv2.putText.call_args_list)
        for call in self.cv2.putText.call_args_list:
            text, (x, y) = call.args[1], call.args[2]
            (tw, th), baseline = fake_text_size(text, None, None, None)
            self.assertGreaterEqual(x, 0, text)
            self.assertLessEqual(x + tw, W, text)
            self.assertGreaterEqual(y - th, 0, text)
            self.assertLessEqual(y + baseline, H, text)


class ThumbFeedbackTest(unittest.TestCase):
    """Raw per-hand thumb-gap readout (OUT/FOLDED/BETWEEN + ratio)."""

    def setUp(self):
        self.cv2 = mock.MagicMock()
        self.cv2.getTextSize.side_effect = fake_text_size
        patcher = mock.patch.dict(sys.modules, {"cv2": self.cv2})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.img = FakeImage()

    def texts(self):
        return [c.args[1] for c in self.cv2.putText.call_args_list]

    def by_text(self):
        return {c.args[1]: c for c in self.cv2.putText.call_args_list}

    def thumb_calls(self):
        return {t: c for t, c in self.by_text().items()
                if re.match(r"^[LR] thumb (OUT|FOLDED|BETWEEN) ", t)}

    @staticmethod
    def gap_of(hand):
        from handcraft.geometry import (INDEX_MCP, THUMB_TIP, dist,
                                        palm_size)
        return dist(hand[THUMB_TIP], hand[INDEX_MCP]) / palm_size(hand)

    def test_thumb_out_green_with_gap_ratio(self):
        hand = make_hand(0.3, 0.5, thumb=(0.1, 0.5, 0.0))  # gap >> .65
        draw_control_overlay(self.img, {"left": hand}, base_state())
        calls = self.thumb_calls()
        (text, call), = calls.items()
        self.assertTrue(text.startswith("L thumb OUT "), text)
        self.assertIn(f"{self.gap_of(hand):.2f}", text)
        self.assertEqual(call.args[5], control_overlay.ACTIVE_COLOR)

    def test_thumb_folded_gray_with_gap_ratio(self):
        hand = make_hand(0.7, 0.5, thumb=(0.75, 0.42, 0.0))  # == index MCP
        draw_control_overlay(self.img, {"right": hand}, base_state())
        calls = self.thumb_calls()
        (text, call), = calls.items()
        self.assertTrue(text.startswith("R thumb FOLDED "), text)
        self.assertIn(f"{self.gap_of(hand):.2f}", text)
        self.assertEqual(call.args[5], control_overlay.NEUTRAL_COLOR)

    def test_thumb_between_yellow(self):
        hand = make_hand(0.3, 0.5, thumb=(0.29, 0.42, 0.0))  # gap ~0.59
        self.assertGreater(self.gap_of(hand), 0.55)
        self.assertLess(self.gap_of(hand), 0.65)
        draw_control_overlay(self.img, {"left": hand}, base_state())
        calls = self.thumb_calls()
        (text, call), = calls.items()
        self.assertTrue(text.startswith("L thumb BETWEEN "), text)
        self.assertEqual(call.args[5], control_overlay.THUMB_BETWEEN_COLOR)

    def test_thumb_threshold_boundaries(self):
        """Strict like thumb_hysteresis: OUT > .65, FOLDED < .55, and
        the exact boundaries (.65/.55) both land in BETWEEN."""
        hand = make_hand()
        cases = ((0.65, "BETWEEN"), (0.651, "OUT"), (0.80, "OUT"),
                 (0.55, "BETWEEN"), (0.549, "FOLDED"), (0.10, "FOLDED"),
                 (0.60, "BETWEEN"))
        for gap, word in cases:
            with mock.patch.object(control_overlay, "_thumb_gap",
                                   return_value=gap):
                got = control_overlay._thumb_feedback(hand)
            self.assertEqual(got[0], word, gap)

    def test_thumb_feedback_uses_shared_helper_when_available(self):
        """If geometry gains thumb_gap_ratio, the overlay prefers it."""
        hand = make_hand()
        with mock.patch.object(control_overlay, "_thumb_gap_ratio",
                               return_value=0.7):
            self.assertEqual(control_overlay._thumb_gap(hand), 0.7)

    def test_thumb_feedback_invalid_or_missing_hand_no_label(self):
        bad = [(float("nan"), 0.5, 0.0)] * 21
        draw_control_overlay(self.img, {"left": bad, "right": None},
                             base_state())
        self.assertEqual(self.thumb_calls(), {})
        draw_control_overlay(self.img, {}, base_state())
        self.assertEqual(self.thumb_calls(), {})

    def test_thumb_feedback_shares_label_rows_without_overlap(self):
        hands = {"left": make_hand(0.3, 0.5, thumb=(0.1, 0.5, 0.0)),
                 "right": make_hand(0.7, 0.5, thumb=(0.75, 0.42, 0.0))}
        draw_control_overlay(self.img, hands,
                             base_state(labels={"left": "Move joystick",
                                                "right": "Look"}))
        calls = self.by_text()
        thumb = self.thumb_calls()
        l_text, l_call = next((t, c) for t, c in thumb.items()
                              if t.startswith("L thumb "))
        r_text, r_call = next((t, c) for t, c in thumb.items()
                              if t.startswith("R thumb "))
        # Same rows as the pose labels, starting after the label text.
        self.assertEqual(l_call.args[2][1], calls["Move joystick"].args[2][1])
        self.assertEqual(r_call.args[2][1], calls["Look"].args[2][1])
        label_tw = fake_text_size("Move joystick", None, None, None)[0][0]
        self.assertEqual(l_call.args[2][0], 6 + label_tw + 14)
        label_tw = fake_text_size("Look", None, None, None)[0][0]
        self.assertEqual(r_call.args[2][0], 6 + label_tw + 14)

    def test_thumb_feedback_without_labels_starts_rows(self):
        hands = {"left": make_hand(0.3, 0.5), "right": make_hand(0.7, 0.5)}
        draw_control_overlay(self.img, hands, base_state())
        thumb = self.thumb_calls()
        self.assertEqual(len(thumb), 2)
        positions = sorted(c.args[2] for c in thumb.values())
        self.assertEqual(positions, [(6, 40), (6, 62)])

    def test_thumb_feedback_drawn_in_inventory_and_hotbar(self):
        """Raw detection feedback is mode-independent; it never claims
        OS input, so it stays visible in every mode."""
        hand = make_hand(0.7, 0.5, thumb=(0.75, 0.42, 0.0))
        for state in (base_state(inventory_open=True),
                      base_state(hotbar_active=True)):
            self.cv2.putText.reset_mock()
            draw_control_overlay(self.img, {"right": hand}, state)
            self.assertEqual(len(self.thumb_calls()), 1)


class ThumbFeedbackRealWidthTest(unittest.TestCase):
    """Layout proof with real cv2 metrics on a 480px-wide preview.

    The mocked suite uses a synthetic getTextSize; here the real font
    metrics verify the worst pose label from posemap plus the longest
    thumb feedback still fit on one row without clamping or overlap.
    """

    # Every pose label posemap can emit for the top rows.
    POSE_LABELS = ("Sneak joystick", "Move joystick",
                   "Hotbar: fold thumb left / pinky right",
                   "Inventory cursor", "Look", "Use / place",
                   "Thumb attack gesture")

    def test_label_plus_thumb_feedback_fit_480px(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("real cv2 not available")

        def width(text):
            return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                   control_overlay._FONT_SCALE,
                                   control_overlay._THICKNESS)[0][0]

        # Longest feedback text over both hands and all three states;
        # the gap ratio is always 4 chars ("0.00"), so width is stable.
        feedback_w = max(
            width(f"{tag} thumb {word} 0.00")
            for tag in "LR" for word in ("OUT", "FOLDED", "BETWEEN"))
        for label in self.POSE_LABELS:
            with self.subTest(label=label):
                # Feedback starts 14px right of the label (no overlap)
                # and must end inside the 480px frame (no clamping).
                x = 6 + width(label) + 14
                self.assertLessEqual(x + feedback_w, 480)


class CameraWorkerOverlayStateTest(unittest.TestCase):
    def test_camera_worker_has_control_overlay_state(self):
        worker = CameraWorker(0, "model", queue.Queue(), queue.Queue(),
                              lambda m: None)
        self.assertEqual(worker.control_overlay_state, {})
        worker.control_overlay_state = {"held_keys": ("W",)}
        self.assertEqual(worker.control_overlay_state["held_keys"], ("W",))

    def test_tracker_draws_overlay_after_landmarks_before_encode(self):
        import inspect
        from handcraft import tracker
        src = inspect.getsource(tracker.CameraWorker._run)
        i_landmarks = src.index("draw_landmark_overlay(rgb, hands)")
        i_control = src.index("draw_control_overlay(")
        i_encode = src.index("self._encode_ppm(rgb)")
        self.assertLess(i_landmarks, i_control)
        self.assertLess(i_control, i_encode)
        self.assertIn("self.control_overlay_state", src)


if __name__ == "__main__":
    unittest.main()
