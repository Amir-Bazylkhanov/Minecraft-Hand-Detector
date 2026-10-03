"""Minecraft jump presses span game updates without blocking the GUI."""
import unittest
from unittest import mock

from handcraft import win32input
from handcraft.adapters import VK_SPACE, VK_W
from handcraft.controller import SafetyController
from handcraft.controller import ArmSequencer
from handcraft.bindings import RecognitionResult
from handcraft.bindings import PalmContactRecognizer
from handcraft.app import HandCraftApp, ResultDispatcher
from handcraft.gestures import GestureEngine


class JumpTimingTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.adapter = win32input.SendInputAdapter(clock=lambda: self.now)
        self.send = mock.patch.object(win32input, '_send', return_value=1)
        self.mock_send = self.send.start()
        self.addCleanup(self.send.stop)

    def test_space_down_and_up_are_separated_by_120ms(self):
        self.adapter.tap_key(VK_SPACE)
        self.assertEqual(self.mock_send.call_count, 1)
        self.assertEqual(self.adapter.held_keys, {VK_SPACE})
        self.assertEqual(self.mock_send.call_args.args[0].ki.dwFlags, 0)
        self.now += 0.119
        self.adapter.service_pending()
        self.assertEqual(self.mock_send.call_count, 1)
        self.now += 0.002
        self.adapter.service_pending()
        self.assertEqual(self.mock_send.call_count, 2)
        self.assertEqual(self.mock_send.call_args.args[0].ki.dwFlags,
                         win32input.KEYEVENTF_KEYUP)
        self.assertFalse(self.adapter.held_keys)

    def test_repeat_does_not_extend_press_and_walk_is_unchanged(self):
        self.adapter.key_down(VK_W)
        self.adapter.tap_key(VK_SPACE)
        self.now += 0.1
        self.adapter.tap_key(VK_SPACE)
        self.now += 0.03
        self.adapter.service_pending()
        self.assertEqual(self.adapter.held_keys, {VK_W})

    def test_release_all_cancels_deadline_and_new_jump_can_fire(self):
        self.adapter.tap_key(VK_SPACE)
        self.adapter.release_all()
        self.assertFalse(self.adapter._pulse_deadlines)
        self.adapter.tap_key(VK_SPACE)
        self.assertEqual(self.adapter.held_keys, {VK_SPACE})

    def test_explicit_space_hold_takes_over_timed_pulse(self):
        self.adapter.tap_key(VK_SPACE)
        self.adapter.key_down(VK_SPACE)
        self.now += 1
        self.adapter.service_pending()
        self.assertEqual(self.adapter.held_keys, {VK_SPACE})

    def test_service_releases_on_focus_loss_before_deadline(self):
        focused = [True]
        ctl = SafetyController(self.adapter, focus_check=lambda: focused[0])
        ctl.set_practice(False)
        ctl.set_armed(True)
        self.adapter.tap_key(VK_SPACE)
        focused[0] = False
        ctl.service_inputs()
        self.assertFalse(ctl.armed)
        self.assertFalse(self.adapter.held_keys)
        # No unrelated physical mouse button was released.
        self.assertTrue(all(call.args[0].type == win32input.INPUT_KEYBOARD
                            for call in self.mock_send.call_args_list))

    def test_app_invalidates_inventory_when_pending_jump_loses_focus(self):
        focused = [True]
        ctl = SafetyController(self.adapter, focus_check=lambda: focused[0])
        ctl.set_practice(False)
        ctl.set_armed(True)
        app = HandCraftApp.__new__(HandCraftApp)
        app.controller = ctl
        app.engine = GestureEngine()
        app.engine.set_inventory_state(False)
        app.dispatcher = ResultDispatcher(app.engine, PalmContactRecognizer(), ctl)
        app.sequencer = ArmSequencer()
        app._focus_achieved = True
        app._focus_ok = True
        self.adapter.tap_key(VK_SPACE)
        focused[0] = False
        app._service_inputs()
        self.assertFalse(ctl.armed)
        self.assertIsNone(app.engine.inventory_believed)
        self.assertFalse(app._focus_achieved)
        self.assertFalse(app._focus_ok)
        self.assertFalse(self.adapter.held_keys)

    def test_movement_blocked_cancels_pending_jump_immediately(self):
        ctl = SafetyController(self.adapter)
        ctl.set_practice(False)
        ctl.set_armed(True)
        self.adapter.tap_key(VK_SPACE)
        result = RecognitionResult(frozenset(), (), {}, {}, {}, frozenset({'left'}))
        ctl.apply_bindings(result, movement_enabled=False)
        self.assertFalse(self.adapter.held_keys)

    def test_failed_release_disarms_and_cleanup_retries(self):
        ctl = SafetyController(self.adapter, focus_check=lambda: True)
        ctl.set_practice(False)
        ctl.set_armed(True)
        self.adapter.tap_key(VK_SPACE)
        self.now += 0.13
        with mock.patch.object(win32input, '_send',
                               side_effect=win32input.SendInputError('failed')):
            with self.assertRaises(win32input.SendInputError):
                ctl.service_inputs()
        self.assertFalse(ctl.armed)
        self.assertIn(VK_SPACE, self.adapter.held_keys)
        ctl.release_all()
        self.assertFalse(self.adapter.held_keys)


if __name__ == '__main__':
    unittest.main()
