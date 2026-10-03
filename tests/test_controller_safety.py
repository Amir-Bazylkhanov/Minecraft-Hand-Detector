"""Controller safety tests: failure paths, focus gating of every live
frame, movement suppression, and mouse-only STOP_ATTACK.

Everything is mock/fake adapters only — no camera, no physical input.
"""

import unittest
from unittest import mock

from handcraft.adapters import RecordingAdapter, VK_E, VK_SPACE, VK_W
from handcraft.bindings import RecognitionResult
from handcraft.controller import SafetyController
from handcraft.gestures import GestureEvent, GestureKind


def result(held=(), pulses=()):
    return RecognitionResult(
        held_keys=frozenset(held),
        pulse_keys=tuple(pulses),
        contact_states={},
        confidences={},
        distances={},
        tracked_hands=frozenset(),
    )


def ev(kind, at=0.0):
    return GestureEvent(kind, at)


class StuckTapAdapter(RecordingAdapter):
    """A tap whose key-up cleanup failed leaves the key adapter-pending."""

    def __init__(self):
        super().__init__()
        self.pending = set()
        self.fail_tap = False

    @property
    def held_keys(self):
        return frozenset(self.pending)

    def tap_key(self, vk):
        self.calls.append(("tap_key", vk))
        if self.fail_tap:
            self.pending.add(vk)  # key-down went out, key-up failed
            raise RuntimeError("tap cleanup failed")

    def release_all(self):
        self.calls.append(("release_all", None))
        for vk in sorted(self.pending):
            self.calls.append(("key_up", vk))
            self.pending.discard(vk)
        if self.left_held:
            self.left_mouse_up()


class FailingMouseAdapter(RecordingAdapter):
    def __init__(self):
        super().__init__()
        self.fail_mouse_up = False

    def left_mouse_up(self):
        self.calls.append(("left_up", None))
        if self.fail_mouse_up:
            raise RuntimeError("mouse-up failed")
        self.left_held = False


class FailingKeyDownAdapter(RecordingAdapter):
    def key_down(self, vk):
        self.calls.append(("key_down", vk))
        raise RuntimeError("send failed")


class FailingKeyUpAdapter(RecordingAdapter):
    def __init__(self):
        super().__init__()
        self.fail_key_up = False

    def key_up(self, vk):
        self.calls.append(("key_up", vk))
        if self.fail_key_up:
            raise RuntimeError("key-up failed")
        self.keys_held.discard(vk)


class ControllerSafetyTests(unittest.TestCase):
    def make_ctl(self, adapter, focus_check=None):
        self.logs = []
        ctl = SafetyController(adapter, on_action=self.logs.append,
                               focus_check=focus_check)
        ctl.set_practice(False)
        return ctl

    def test_pending_adapter_tap_key_retried_by_release_all(self):
        adapter = StuckTapAdapter()
        adapter.fail_tap = True
        ctl = self.make_ctl(adapter)
        ctl.set_armed(True)

        with self.assertRaises(RuntimeError):
            ctl.apply_bindings(result(pulses=["E"]))

        self.assertFalse(ctl.armed, "input failure must disarm")
        self.assertIn(("release_all", None), adapter.calls,
                      "release_all must retry adapter-only pending tap keys")
        self.assertEqual(adapter.pending, set())

    def test_mouse_release_failure_still_cleans_keys_and_disarms(self):
        adapter = FailingMouseAdapter()
        ctl = self.make_ctl(adapter)
        ctl.set_armed(True)
        ctl.apply_bindings(result(held=["W"]))
        ctl.handle([ev(GestureKind.PUNCH)])
        self.assertIn(("key_down", VK_W), adapter.calls)
        self.assertIn(("left_down", None), adapter.calls)

        adapter.fail_mouse_up = True
        ctl.set_armed(False)  # must not raise

        self.assertFalse(ctl.armed)
        self.assertIn(("left_up", None), adapter.calls,
                      "mouse release must be attempted")
        self.assertIn(("key_up", VK_W), adapter.calls,
                      "key cleanup must proceed despite mouse failure")
        self.assertEqual(ctl.held_keys, set())
        self.assertTrue(ctl._os_held,
                        "failed mouse release keeps its bookkeeping")
        self.assertFalse(ctl.mouse_held)

    def test_unchanged_held_key_is_gated_every_frame(self):
        adapter = RecordingAdapter()
        focus = mock.Mock(side_effect=[True, True, False])
        ctl = self.make_ctl(adapter, focus_check=focus)
        ctl.set_armed(True)

        ctl.apply_bindings(result(held=["W"]))
        self.assertIn(("key_down", VK_W), adapter.calls)

        # Identical frame: no new key to press, but the gate still runs.
        ctl.apply_bindings(result(held=["W"]))

        self.assertEqual(focus.call_count, 3,
                         "every live frame and every dispatch must be "
                         "focus-checked")
        self.assertIn(("key_up", VK_W), adapter.calls,
                      "focus loss must release the held key")
        self.assertFalse(ctl.armed, "focus loss must disarm")

    def test_inventory_suppresses_pulses_and_movement_holds(self):
        adapter = RecordingAdapter()
        ctl = self.make_ctl(adapter)
        ctl.set_armed(True)
        ctl.apply_bindings(result(held=["W"]))
        self.assertIn(("key_down", VK_W), adapter.calls)

        ctl.apply_bindings(result(held=["W"], pulses=["SPACE"]),
                           movement_enabled=False)

        taps = [c for c in adapter.calls if c[0] == "tap_key"]
        self.assertNotIn(("tap_key", VK_SPACE), taps,
                         "inventory-open frames must not pulse Space")
        self.assertIn(("key_up", VK_W), adapter.calls,
                      "held movement keys must be released")
        self.assertEqual(ctl.held_keys, set())

    def test_stop_attack_releases_mouse_but_keeps_movement_hold(self):
        adapter = RecordingAdapter()
        ctl = self.make_ctl(adapter)
        ctl.set_armed(True)
        ctl.apply_bindings(result(held=["W"]))
        ctl.handle([ev(GestureKind.PUNCH)])

        ctl.handle([ev(GestureKind.STOP_ATTACK)])

        kinds = adapter.kinds()
        self.assertIn("left_up", kinds)
        self.assertNotIn(("key_up", VK_W), adapter.calls,
                         "STOP_ATTACK must not release held movement keys")
        self.assertIn(VK_W, ctl.held_keys)
        self.assertFalse(ctl.mouse_held)

    def test_stop_attack_release_failure_disarms_and_reraises(self):
        adapter = FailingMouseAdapter()
        ctl = self.make_ctl(adapter)
        ctl.set_armed(True)
        ctl.handle([ev(GestureKind.PUNCH)])
        self.assertIn(("left_down", None), adapter.calls)

        adapter.fail_mouse_up = True
        with self.assertRaises(RuntimeError):
            ctl.handle([ev(GestureKind.STOP_ATTACK)])

        self.assertFalse(ctl.armed, "release failure must disarm")
        self.assertTrue(ctl._os_held,
                        "failed mouse release keeps its bookkeeping")
        self.assertFalse(ctl.mouse_held)

        # Retry path: once the adapter recovers, a repeated STOP_ATTACK
        # releases the still-held OS button even though _held is False.
        adapter.fail_mouse_up = False
        ctl.handle([ev(GestureKind.STOP_ATTACK)])
        self.assertFalse(ctl._os_held)
        # 1 (STOP) + 2 (fail-closed release_all + adapter release_all
        # retries) + 1 (successful retry) attempts.
        self.assertEqual(adapter.kinds().count("left_up"), 4)

    def test_keyup_contact_end_failure_disarms_and_reraises(self):
        adapter = FailingKeyUpAdapter()
        ctl = self.make_ctl(adapter)
        ctl.set_armed(True)
        ctl.apply_bindings(result(held=["W"]))
        self.assertIn(("key_down", VK_W), adapter.calls)

        adapter.fail_key_up = True
        with self.assertRaises(RuntimeError) as ctx:
            ctl.apply_bindings(result())  # contact ended -> key_up fails

        self.assertEqual(str(ctx.exception), "key-up failed",
                         "the original error must propagate")
        self.assertFalse(ctl.armed, "release failure must disarm")
        self.assertIn(VK_W, ctl.held_keys,
                      "failed key release stays tracked for retry")

    def test_focus_loss_at_tap_gate_blocks_pulse_and_releases_hold(self):
        adapter = RecordingAdapter()
        # Frame gate OK, hold gate OK, then focus lost right before the tap.
        focus = mock.Mock(side_effect=[True, True, False])
        ctl = self.make_ctl(adapter, focus_check=focus)
        ctl.set_armed(True)

        ctl.apply_bindings(result(held=["W"], pulses=["SPACE"]))

        self.assertNotIn(("tap_key", VK_SPACE), adapter.calls,
                         "focus loss before a tap must suppress the pulse")
        self.assertIn(("key_down", VK_W), adapter.calls)
        self.assertIn(("key_up", VK_W), adapter.calls,
                      "focus loss must release the held key")
        self.assertFalse(ctl.armed, "focus loss must disarm")
        self.assertEqual(focus.call_count, 3,
                         "gate runs per frame and per dispatch")

    def test_pulse_matching_held_key_is_skipped(self):
        adapter = RecordingAdapter()
        ctl = self.make_ctl(adapter)
        ctl.set_armed(True)

        ctl.apply_bindings(result(held=["W"], pulses=["W"]))

        self.assertIn(("key_down", VK_W), adapter.calls)
        self.assertNotIn(("tap_key", VK_W), adapter.calls,
                         "a tap's key-up must never interrupt a held key")
        self.assertIn(VK_W, ctl.held_keys)
        self.assertTrue(ctl.armed)

    def test_input_failure_disarms_cleans_up_and_reraises(self):
        adapter = FailingKeyDownAdapter()
        ctl = self.make_ctl(adapter)
        ctl.set_armed(True)
        ctl.handle([ev(GestureKind.PUNCH)])  # mouse held
        self.assertIn(("left_down", None), adapter.calls)

        with self.assertRaises(RuntimeError) as ctx:
            ctl.apply_bindings(result(held=["W"]))

        self.assertEqual(str(ctx.exception), "send failed",
                         "the original error must propagate")
        self.assertFalse(ctl.armed, "failure must disarm before cleanup")
        self.assertIn(("left_up", None), adapter.calls,
                      "cleanup must release the held mouse")
        self.assertFalse(ctl._os_held)

    def test_practice_mode_sends_nothing_to_os(self):
        adapter = RecordingAdapter()
        ctl = self.make_ctl(adapter)
        ctl.set_practice(True)
        ctl.set_armed(True)

        ctl.apply_bindings(result(held=["W"], pulses=["SPACE"]))
        ctl.handle([ev(GestureKind.PUNCH), ev(GestureKind.OPEN_INVENTORY),
                    ev(GestureKind.STOP_ATTACK)])
        ctl.release_all("test")

        self.assertEqual(adapter.calls, [])
        self.assertTrue(any("[practice]" in m for m in self.logs))

    def test_focus_check_none_stays_permissive(self):
        adapter = RecordingAdapter()
        ctl = self.make_ctl(adapter, focus_check=None)
        ctl.set_armed(True)
        ctl.apply_bindings(result(held=["W"], pulses=["SPACE"]))
        self.assertIn(("key_down", VK_W), adapter.calls)
        self.assertIn(("tap_key", VK_SPACE), adapter.calls)
        self.assertTrue(ctl.armed)

    def test_release_all_without_pending_adapter_state_skips_adapter_call(self):
        # RecordingAdapter exposes no adapter-only pending state, so a plain
        # release_all must not touch the adapter beyond our own holds.
        adapter = RecordingAdapter()
        ctl = self.make_ctl(adapter)
        ctl.set_armed(True)
        ctl.release_all("test")
        self.assertEqual(adapter.calls, [])


if __name__ == "__main__":
    unittest.main()
