import unittest

from handcraft.adapters import RecordingAdapter, VK_E
from handcraft.controller import SafetyController
from handcraft.gestures import GestureEvent, GestureKind


def ev(kind, at=0.0):
    return GestureEvent(kind, at)


class TestSafetyController(unittest.TestCase):
    def setUp(self):
        self.adapter = RecordingAdapter()
        self.logs = []
        self.ctl = SafetyController(self.adapter, on_action=self.logs.append)
        self.ctl.set_practice(False)

    def test_disarmed_blocks_injection(self):
        self.ctl.handle([ev(GestureKind.PUNCH), ev(GestureKind.OPEN_INVENTORY)])
        self.assertEqual(self.adapter.calls, [])

    def test_punch_is_a_hold_never_a_click(self):
        self.ctl.set_armed(True)
        self.ctl.handle([ev(GestureKind.PUNCH), ev(GestureKind.PUNCH),
                         ev(GestureKind.PUNCH)])
        downs = [c for c in self.adapter.calls if c[0] == "left_down"]
        self.assertEqual(len(downs), 1, "repeated punches must hold, not click")

    def test_stop_attack_releases(self):
        self.ctl.set_armed(True)
        self.ctl.handle([ev(GestureKind.PUNCH)])
        self.ctl.handle([ev(GestureKind.STOP_ATTACK)])
        self.assertEqual(self.adapter.kinds(), ["left_down", "left_up"])

    def test_stop_attack_honoured_while_disarmed(self):
        self.ctl.set_armed(True)
        self.ctl.handle([ev(GestureKind.PUNCH)])
        self.ctl.set_armed(False)  # disarm releases
        self.assertIn("left_up", self.adapter.kinds())
        # A stray STOP_ATTACK while disarmed is safe and idempotent.
        self.ctl.handle([ev(GestureKind.STOP_ATTACK)])
        self.assertEqual(self.adapter.kinds().count("left_up"), 1)

    def test_inventory_taps_e_once(self):
        self.ctl.set_armed(True)
        self.ctl.handle([ev(GestureKind.OPEN_INVENTORY)])
        self.ctl.handle([ev(GestureKind.CLOSE_INVENTORY)])
        taps = [c for c in self.adapter.calls if c[0] == "tap_key"]
        self.assertEqual(taps, [("tap_key", VK_E), ("tap_key", VK_E)])

    def test_practice_mode_never_touches_adapter(self):
        self.ctl.set_practice(True)
        self.ctl.set_armed(True)
        self.ctl.handle([ev(GestureKind.PUNCH), ev(GestureKind.OPEN_INVENTORY),
                         ev(GestureKind.STOP_ATTACK)])
        self.assertEqual(self.adapter.calls, [])
        self.assertTrue(any("[practice]" in m for m in self.logs))

    def test_release_all_only_releases_what_we_pressed(self):
        # In practice mode a hold is logical only; release_all must not
        # emit a real OS-level mouse-up.
        self.ctl.set_practice(True)
        self.ctl.set_armed(True)
        self.ctl.handle([ev(GestureKind.PUNCH)])
        self.ctl.release_all("test")
        self.assertEqual(self.adapter.calls, [])

    def test_disarm_releases_held_mouse(self):
        self.ctl.set_armed(True)
        self.ctl.handle([ev(GestureKind.PUNCH)])
        self.ctl.set_armed(False)
        self.assertEqual(self.adapter.kinds(), ["left_down", "left_up"])
        self.assertFalse(self.ctl.mouse_held)


if __name__ == "__main__":
    unittest.main()
