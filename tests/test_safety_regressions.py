"""Safety regression tests for the gesture engine and the Win32 adapter.

Everything here is mock-driven: no camera, no real SendInput, no clock
reads. The Win32 tests patch ``handcraft.win32input._send`` (or the
module's ``_user32`` handle), so nothing is ever physically injected.
"""

import unittest
from unittest import mock

from handcraft import win32input
from handcraft.gestures import GestureConfig, GestureEngine, GestureKind
from tests.hands import make_hand, moved

VK_E = 0x45


def kinds(events):
    return [e.kind for e in events]


class GestureSafetyRegressionTests(unittest.TestCase):
    def setUp(self):
        self.engine = GestureEngine(GestureConfig())
        self.fist = make_hand("fist")

    def _punch(self):
        """Acquire, dwell, then jab: ends with the attack hold active."""
        self.engine.update(self.fist, 0.0)                 # acquire
        self.engine.update(self.fist, 0.4)                 # acquire dwell done
        self.engine.update(self.fist, 0.50)                # still
        events = self.engine.update(moved(self.fist, dx=1.0), 0.55)  # jab
        self.assertEqual(kinds(events), [GestureKind.PUNCH])
        self.assertTrue(self.engine.attacking)

    def test_open_palm_releases_attack_immediately_and_before_open_event(self):
        self._punch()

        # The palm opens: the attack must die on this very frame, long
        # before the open-palm dwell could produce OPEN_INVENTORY.
        palm = moved(make_hand("open"), dx=1.0)
        step = self.engine.update(palm, 0.60)
        self.assertIn(GestureKind.STOP_ATTACK, kinds(step))
        self.assertNotIn(GestureKind.OPEN_INVENTORY, kinds(step))
        self.assertFalse(self.engine.attacking)

        # Once the palm is stable, OPEN_INVENTORY fires — after STOP_ATTACK.
        events = step + self.engine.update(palm, 1.10)
        self.assertIn(GestureKind.OPEN_INVENTORY, kinds(events))
        self.assertLess(kinds(events).index(GestureKind.STOP_ATTACK),
                        kinds(events).index(GestureKind.OPEN_INVENTORY))

    def test_stop_attack_precedes_close_inventory(self):
        self._punch()

        # Manual sync: the game inventory turns out to be open. A stable
        # fist now closes it, and the attack release must come first.
        self.engine.set_inventory_state(True)
        events = self.engine.update(moved(self.fist, dx=1.0), 0.60)
        self.assertIn(GestureKind.CLOSE_INVENTORY, kinds(events))
        self.assertIn(GestureKind.STOP_ATTACK, kinds(events))
        self.assertLess(kinds(events).index(GestureKind.STOP_ATTACK),
                        kinds(events).index(GestureKind.CLOSE_INVENTORY))

    def test_fast_motion_cannot_restart_after_lease_expiry_without_rearm(self):
        self._punch()  # punch at 0.55, lease expires at 1.35

        # Uninterrupted fast motion: no return stroke ever rearms the
        # engine, and lease expiry must not rearm it either.
        t = 0.60
        x = 2.0
        expired = False
        post_expiry = []
        while t <= 1.50:
            events = self.engine.update(moved(self.fist, dx=x), t)
            if GestureKind.STOP_ATTACK in kinds(events):
                expired = True
            elif expired:
                post_expiry.extend(events)
            t += 0.05
            x += 1.0
        self.assertTrue(expired, "lease should have expired")
        self.assertFalse(self.engine.attacking)
        self.assertNotIn(GestureKind.PUNCH, kinds(post_expiry),
                         "fast motion alone must not restart mining after "
                         "lease expiry without a rearm")

        # Slow down below the rearm threshold...
        self.engine.update(moved(self.fist, dx=x), 1.55)
        self.engine.update(moved(self.fist, dx=x), 1.60)   # velocity ~0: rearm
        # ...and the next jab mines again.
        events = self.engine.update(moved(self.fist, dx=x + 1.0), 1.65)
        self.assertEqual(kinds(events), [GestureKind.PUNCH])
        self.assertTrue(self.engine.attacking)


class Win32InputSafetyTests(unittest.TestCase):
    """All sends are mocked; no input is ever physically injected."""

    def setUp(self):
        self.adapter = win32input.SendInputAdapter()

    def test_send_raises_on_failed_or_partial_send(self):
        with mock.patch.object(win32input, "_user32") as user32:
            user32.SendInput.return_value = 0  # total failure
            with self.assertRaises(win32input.SendInputError):
                win32input._send(
                    win32input._mouse_event(win32input.MOUSEEVENTF_LEFTDOWN))
            user32.SendInput.return_value = 1  # partial: 1 of 2 injected
            with self.assertRaises(win32input.SendInputError):
                win32input._send(
                    win32input._key_event(VK_E, 0),
                    win32input._key_event(VK_E, win32input.KEYEVENTF_KEYUP))
            user32.SendInput.return_value = 2  # full batch
            sent = win32input._send(
                win32input._key_event(VK_E, 0),
                win32input._key_event(VK_E, win32input.KEYEVENTF_KEYUP))
            self.assertEqual(sent, 2)

    def test_tap_key_attempts_keyup_cleanup_on_failure(self):
        calls = []

        def fake_send(*inputs):
            calls.append(inputs)
            if len(calls) == 1:
                raise win32input.SendInputError("boom")
            return len(inputs)

        with mock.patch.object(win32input, "_send", side_effect=fake_send):
            with self.assertRaises(win32input.SendInputError):
                self.adapter.tap_key(VK_E)

        self.assertEqual(len(calls), 2, "a cleanup key-up must be attempted")
        cleanup = calls[1]
        self.assertEqual(len(cleanup), 1)
        self.assertEqual(cleanup[0].ki.wVk, VK_E)
        self.assertTrue(cleanup[0].ki.dwFlags & win32input.KEYEVENTF_KEYUP)
        self.assertEqual(self.adapter.held_keys, frozenset(),
                         "successful cleanup leaves nothing tracked")

    def test_tap_key_double_failure_tracks_key_and_release_all_clears_it(self):
        with mock.patch.object(win32input, "_send",
                               side_effect=win32input.SendInputError("boom")):
            with self.assertRaises(win32input.SendInputError):
                self.adapter.tap_key(VK_E)
        self.assertEqual(self.adapter.held_keys, frozenset({VK_E}),
                         "an uncertain stuck key must stay tracked")

        with mock.patch.object(win32input, "_send", return_value=1) as send:
            self.adapter.release_all()
        self.assertEqual(send.call_count, 1, "release only the tracked key, not an unowned mouse")
        self.assertEqual(self.adapter.held_keys, frozenset())

    def test_key_down_up_are_idempotent_and_track_held_keys(self):
        with mock.patch.object(win32input, "_send", return_value=1) as send:
            self.adapter.key_down(VK_E)
            self.adapter.key_down(VK_E)  # idempotent: held, not re-pressed
            self.assertEqual(send.call_count, 1)
            self.assertEqual(self.adapter.held_keys, frozenset({VK_E}))

            self.adapter.key_up(VK_E)
            self.assertEqual(send.call_count, 2)
            self.assertEqual(self.adapter.held_keys, frozenset())

            self.adapter.key_up(VK_E)  # idempotent: not held, nothing sent
            self.assertEqual(send.call_count, 2)

    def test_failed_key_down_is_not_tracked(self):
        with mock.patch.object(win32input, "_send",
                               side_effect=win32input.SendInputError("boom")):
            with self.assertRaises(win32input.SendInputError):
                self.adapter.key_down(VK_E)
        self.assertEqual(self.adapter.held_keys, frozenset())

    def test_release_all_attempts_everything_and_preserves_failed_holds(self):
        with mock.patch.object(win32input, "_send", return_value=1):
            self.adapter.key_down(VK_E)
            self.adapter.left_mouse_down()

        calls = []

        def flaky_send(*inputs):
            calls.append(inputs)
            if len(calls) == 1:
                raise win32input.SendInputError("key-up failed")
            return len(inputs)

        with mock.patch.object(win32input, "_send", side_effect=flaky_send):
            with self.assertRaises(win32input.SendInputError):
                self.adapter.release_all()

        self.assertEqual(len(calls), 2,
                         "mouse release must be attempted even after the "
                         "key release failed")
        self.assertEqual(self.adapter.held_keys, frozenset({VK_E}),
                         "a failed key release must stay tracked as held")
        self.assertFalse(self.adapter._left_held,
                         "the successful mouse release clears its held state")


if __name__ == "__main__":
    unittest.main()
