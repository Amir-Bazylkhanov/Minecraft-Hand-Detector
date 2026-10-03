"""A pending adapter-owned timed click must be up before the menu E tap.

A thumb extend->fold cycle maps to ``click_mouse`` on the adapter, which
holds the button for a timed ~100 ms pulse. If the both-palms inventory
chord lands inside that window, the E tap must not reach the game while
the click button is still down — the inventory would open with
attack/use held. All sends are mocked: no camera, no real input.
"""
import unittest
from unittest import mock

from handcraft import win32input
from handcraft.adapters import RecordingAdapter, VK_E
from handcraft.bindings import PalmContactRecognizer
from handcraft.controller import SafetyController
from handcraft.gestures import GestureEngine, GestureEvent, GestureKind
from handcraft.posedispatch import PoseDispatcher
from handcraft.posemap import PoseResult
from tests.test_posemap import pose


def e_tap_index(send):
    """Position of the first E tap batch in the mocked _send calls."""
    for i, call in enumerate(send.call_args_list):
        inputs = call.args
        if len(inputs) == 2 and all(inp.ki.wVk == VK_E for inp in inputs):
            return i
    return -1


class MenuToggleClickCleanupTests(unittest.TestCase):
    def setUp(self):
        self.now = [0.0]
        self.adapter = win32input.SendInputAdapter(clock=lambda: self.now[0])
        self.ctl = SafetyController(self.adapter, focus_check=lambda: True)
        self.ctl.set_practice(False)
        self.ctl.set_armed(True)
        self.engine = GestureEngine()
        self.dispatch = PoseDispatcher(
            self.engine, PalmContactRecognizer(), self.ctl)
        self.palms = {'left': pose((0, 1, 2, 3), True),
                      'right': pose((0, 1, 2, 3), True)}

    def feed(self, hands, t):
        self.now[0] = t
        self.dispatch.dispatch(hands, t, t)

    def test_pending_thumb_click_released_before_inventory_e(self):
        """Gameplay click: right thumb extend->fold with the fist curled."""
        with mock.patch.object(win32input, '_send', return_value=1) as send:
            self.feed({'right': pose((), True)}, .45)  # thumb extends
            self.feed({'right': pose()}, .5)  # thumb folds -> timed LMB
            self.assertTrue(self.adapter.left_held)
            self.assertEqual(send.call_count, 1)  # one down, still pending

            # First both-open frame lands 40 ms into the 100 ms click.
            self.feed(self.palms, .54)
            self.assertFalse(self.adapter.left_held)
            self.assertTrue(self.engine.inventory_believed)

        calls = send.call_args_list
        down = calls[0].args[0].mi.dwFlags
        self.assertEqual(down, win32input.MOUSEEVENTF_LEFTDOWN)
        up = next(i for i, c in enumerate(calls)
                  if c.args[0].mi.dwFlags == win32input.MOUSEEVENTF_LEFTUP)
        tap = e_tap_index(send)
        self.assertGreaterEqual(tap, 0, "the inventory E tap must still fire")
        self.assertLess(up, tap, "LMB must be up before the E tap is sent")

    def test_pending_use_click_released_before_inventory_e(self):
        with mock.patch.object(win32input, '_send', return_value=1) as send:
            self.feed({'right': pose((0, 1))}, 0)
            self.feed({'right': pose((0, 1))}, .15)  # stable v -> timed RMB
            self.assertTrue(self.adapter.right_held)

            self.feed(self.palms, .19)  # 40 ms into the 100 ms click
            self.assertFalse(self.adapter.right_held)
            self.assertTrue(self.engine.inventory_believed)

        calls = send.call_args_list
        up = next(i for i, c in enumerate(calls)
                  if c.args[0].mi.dwFlags == win32input.MOUSEEVENTF_RIGHTUP)
        tap = e_tap_index(send)
        self.assertGreaterEqual(tap, 0, "the inventory E tap must still fire")
        self.assertLess(up, tap, "RMB must be up before the E tap is sent")

    def test_inventory_point_thumb_click_survives_until_own_deadline(self):
        """A valid inventory click is not cancelled by the next menu frame.

        Only a menu-toggle frame cancels pending clicks; plain frames while
        the inventory is open must leave the click held until the adapter's
        own 100 ms service releases it. The inventory click is a thumb
        extend->fold on the pointing cursor hand; V never clicks here."""
        with mock.patch.object(win32input, '_send', return_value=1) as send:
            # The inventory cursor would call real Win32 window APIs.
            with mock.patch.object(self.adapter, 'move_cursor'):
                self.feed(self.palms, 0)  # open the menu first
                self.assertTrue(self.engine.inventory_believed)
                self.feed({'right': pose((0,), True)}, .03)  # point, thumb out
                # thumb folds on the point -> click, deadline .25
                self.feed({'right': pose((0,))}, .15)
                self.assertTrue(self.adapter.left_held)
                self.feed({'right': pose((0,))}, .18)  # must NOT cancel early
                self.assertTrue(self.adapter.left_held)
            self.now[0] = .25
            self.adapter.service_pending()  # the timed pulse ends on its own
            self.assertFalse(self.adapter.left_held)

        flags = [c.args[0].mi.dwFlags for c in send.call_args_list
                 if len(c.args) == 1 and c.args[0].type == win32input.INPUT_MOUSE]
        self.assertEqual(flags, [win32input.MOUSEEVENTF_LEFTDOWN,
                                 win32input.MOUSEEVENTF_LEFTUP])

    def test_held_mining_released_by_stop_before_inventory_e(self):
        """A controller-held mining lease dies via STOP_ATTACK before E.

        The hold here comes from the parallel classic event writer
        (PUNCH straight into the controller), and the pose frame is a
        hybrid writer result carrying STOP_ATTACK + OPEN_INVENTORY; the
        dispatcher must honour the release before the menu tap."""
        adapter = RecordingAdapter()
        ctl = SafetyController(adapter, focus_check=lambda: True)
        ctl.set_practice(False)
        ctl.set_armed(True)
        engine = GestureEngine()
        dispatch = PoseDispatcher(engine, PalmContactRecognizer(), ctl)
        ctl.handle([GestureEvent(GestureKind.PUNCH, 0.)])
        self.assertTrue(ctl.mouse_held)

        hybrid = PoseResult(events=(
            GestureEvent(GestureKind.STOP_ATTACK, .5),
            GestureEvent(GestureKind.OPEN_INVENTORY, .5)))
        with mock.patch.object(dispatch.pose_engine, 'update',
                               return_value=hybrid):
            dispatch.dispatch(self.palms, .5, .5)

        self.assertFalse(ctl.mouse_held)
        up = adapter.calls.index(('left_up', None))
        tap = adapter.calls.index(('tap_key', VK_E))
        self.assertLess(up, tap)
        self.assertTrue(engine.inventory_believed)


if __name__ == '__main__':
    unittest.main()
