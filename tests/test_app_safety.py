"""App-level safety tests: GUI integration paths of handcraft.app.

Everything runs with a mocked Tk root/widgets, a RecordingAdapter and a
settable focus callable — no camera is opened, no model is downloaded, no
real Tk window is created, and no real OS input is ever sent.
"""

import queue
import tempfile
import threading
import time
import tkinter as tk
import unittest
from pathlib import Path
from unittest import mock

from handcraft import handmap
from handcraft.adapters import RecordingAdapter, VK_W
from handcraft.app import _STATUS_WORKER_MSG, HandCraftApp, ResultDispatcher
from handcraft.bindings import PalmContactRecognizer
from handcraft.controller import ArmSequencer, SafetyController
from handcraft.gestures import GestureEngine
from tests.hands import make_hand


def contact_hand(index_distance=1.4, thumb_distance=1.4):
    """Synthetic 21-landmark hand; distances are palm-normalized."""
    base = [(1.2, -1.2, 0.0) for _ in range(21)]
    base[0] = (0.0, 0.0, 0.0)
    base[5] = (-0.45, -0.75, 0.0)
    base[9] = (0.0, -1.0, 0.0)
    base[13] = (0.4, -0.85, 0.0)
    base[17] = (0.7, -0.7, 0.0)
    palm_points = [base[i] for i in (0, 5, 9, 13, 17)]
    palm_x = sum(p[0] for p in palm_points) / 5
    palm_y = sum(p[1] for p in palm_points) / 5
    scale = (sum((p[0] - palm_x) ** 2 + (p[1] - palm_y) ** 2
                 for p in palm_points) / 5) ** 0.5
    base[8] = (palm_x, palm_y - index_distance * scale, 0.0)
    base[4] = (palm_x - thumb_distance * scale, palm_y, 0.0)
    return base


class _Var:
    """Stand-in for tk.BooleanVar/StringVar without a Tk root."""

    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


def make_app(practice=True, focused=True, config_path=None):
    """HandCraftApp shell with mocked Tk: real engine/recognizer/controller/
    dispatcher wiring, no widgets, no mainloop."""
    app = object.__new__(HandCraftApp)
    app.root = mock.MagicMock(name="root")
    app.log_lines = []
    app.status_lines = []
    app._log_action = app.log_lines.append
    app._set_status = lambda level, text: app.status_lines.append((level, text))

    app.engine = GestureEngine()
    app.recognizer = PalmContactRecognizer()
    app.adapter = RecordingAdapter()
    app.focus = {"focused": focused}
    app.controller = SafetyController(
        app.adapter,
        on_action=app.log_lines.append,
        focus_check=lambda: app.focus["focused"])
    app.controller.practice = practice
    app.dispatcher = ResultDispatcher(
        app.engine, app.recognizer, app.controller,
        gate=app._gestures_live_enabled)
    app.sequencer = ArmSequencer()

    app.frame_queue = queue.Queue(maxsize=2)
    app.result_queue = queue.Queue(maxsize=4)
    app.status_queue = queue.Queue()

    app.worker = None
    app._generation = 0
    app._prep_stop = None
    app._preparing = False
    app._last_capture_at = None
    app._latest_hands = {}
    app._focus_ok = focused
    app._focus_achieved = False

    app.practice_var = _Var(practice)
    app.camera_var = _Var("0")
    app.start_btn = mock.MagicMock()
    app.stop_btn = mock.MagicMock()
    app.inv_label = mock.MagicMock()
    app.preview = mock.MagicMock()

    app._handmap = handmap
    app._config_path = config_path or Path("nonexistent_test_bindings.json")
    app._config_panel = None
    app._config_mtime = None
    return app


class PracticeLiveSwitchTests(unittest.TestCase):
    def test_practice_to_live_disarms_and_cancels(self):
        app = make_app(practice=True)
        app._on_practice()
        app._on_arm()
        self.assertTrue(app.controller.armed)
        self.assertTrue(app.controller.practice)

        app.practice_var.set(False)  # user unchecks practice
        app._on_practice()
        self.assertFalse(app.controller.armed,
                         "mode switch must disarm before going live")
        self.assertFalse(app.controller.practice)
        self.assertFalse(app.sequencer.counting)
        self.assertIsNone(app.engine.inventory_believed,
                          "mode switch resets belief to UNKNOWN")

    def test_live_countdown_cancelled_by_practice_toggle(self):
        app = make_app(practice=False)
        app.engine.set_inventory_state(False)
        app._on_arm()
        self.assertTrue(app.sequencer.counting)

        app.practice_var.set(True)  # user re-checks practice mid-countdown
        app._on_practice()
        self.assertFalse(app.sequencer.counting,
                         "practice toggle must cancel the live countdown")
        self.assertFalse(app.controller.armed)
        self.assertTrue(app.controller.practice)


class LiveArmInventoryTests(unittest.TestCase):
    def test_known_inventory_sync_survives_arm(self):
        app = make_app(practice=False)
        app.engine.set_inventory_state(True)  # manual "Mark open" sync
        app._on_arm()
        self.assertTrue(app.sequencer.counting)
        self.assertFalse(app.controller.armed, "countdown not elapsed yet")
        self.assertIs(app.engine.inventory_believed, True,
                      "known belief must survive the arming reset")
        self.assertTrue(app.sequencer.poll(time.monotonic() + 3.1),
                        "3s countdown completes")

    def test_known_closed_belief_survives_arm(self):
        app = make_app(practice=False)
        app.engine.set_inventory_state(False)  # manual "Mark closed" sync
        app._on_arm()
        self.assertTrue(app.sequencer.counting)
        self.assertIs(app.engine.inventory_believed, False)

    def test_live_arm_refused_when_inventory_unknown(self):
        app = make_app(practice=False)
        app.engine.set_inventory_state(None)
        app._on_arm()
        self.assertFalse(app.sequencer.counting,
                         "no countdown while belief is UNKNOWN")
        self.assertFalse(app.controller.armed)
        self.assertTrue(any("UNKNOWN" in text
                            for _, text in app.status_lines))


class LiveDisarmedGatingTests(unittest.TestCase):
    def test_live_disarmed_hands_do_not_change_belief(self):
        app = make_app(practice=False, focused=True)
        app.engine.set_inventory_state(False)
        self.assertFalse(app.controller.armed)
        open_palm = {"right": make_hand("open")}
        t0 = 1000.0
        for i in range(40):  # ~2s of stable open palm: would open inventory
            t = t0 + i * 0.05
            self.assertTrue(app.dispatcher.dispatch(open_palm, t, t))
        self.assertIs(app.engine.inventory_believed, False,
                      "disarmed live gestures must not set belief without E")
        self.assertEqual(app.adapter.calls, [])
        self.assertIsNotNone(app.dispatcher.last_result,
                             "recognizer still updates for the UI")

    def test_unknown_inventory_blocks_armed_live_actions(self):
        app = make_app(practice=False, focused=True)
        app.engine.set_inventory_state(None)
        app.controller.set_armed(True)
        open_palm = {"right": make_hand("open")}
        t0 = 2000.0
        for i in range(40):
            t = t0 + i * 0.05
            app.dispatcher.dispatch(open_palm, t, t)
        self.assertIsNone(app.engine.inventory_believed,
                          "belief stays UNKNOWN until a manual sync")
        self.assertEqual(app.adapter.calls, [],
                         "no input while inventory belief is unknown")


class StalePacketTests(unittest.TestCase):
    def test_stale_packet_releases_held_w(self):
        app = make_app(practice=False, focused=True)
        app.engine.set_inventory_state(False)
        app.controller.set_armed(True)
        app.worker = mock.MagicMock()

        now = time.monotonic()
        contact = {"Left": contact_hand(index_distance=0.10)}
        app.result_queue.put((contact, now))
        app._tick_body()
        self.assertIn(("key_down", VK_W), app.adapter.calls)

        # Same contact, but captured over 300 ms ago: rejected, and the
        # rejection releases the held W via tracking-loss.
        app.result_queue.put((contact, now - 1.0))
        app._tick_body()
        self.assertIn(("key_up", VK_W), app.adapter.calls)
        self.assertEqual(app.controller.held_keys, set())


class ConfigPanelTests(unittest.TestCase):
    def test_panel_uses_same_tk_root_without_monkeypatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = make_app(config_path=Path(tmp) / "bindings.json")
            created = {}

            class FakePanel:
                def __init__(self, parent=None, config_path=None,
                             on_bindings_changed=None):
                    created["parent"] = parent
                    created["config_path"] = config_path
                    created["callback"] = on_bindings_changed
                    self.destroyed = False

                def winfo_exists(self):
                    return True

                def lift(self):
                    created["lifted"] = True

                def focus_force(self):
                    pass

                def protocol(self, *args, **kwargs):
                    pass

                def destroy(self):
                    self.destroyed = True

            original_tk_init = tk.Tk.__init__
            with mock.patch.object(handmap, "GestureConfigPanel", FakePanel):
                app._open_config_panel()
                self.assertIs(created["parent"], app.root,
                              "panel must attach to the app's Tk root")
                self.assertEqual(created["config_path"], app._config_path)
                self.assertTrue(callable(created["callback"]))
                self.assertIs(tk.Tk.__init__, original_tk_init,
                              "tk.Tk must never be monkeypatched")

                # Reopening lifts the existing panel instead of a new one.
                created.clear()
                app._open_config_panel()
                self.assertTrue(created.get("lifted"))
                self.assertNotIn("parent", created, "no second panel created")
            self.assertIs(tk.Tk.__init__, original_tk_init)

            # The save callback releases held input and reloads bindings.
            with mock.patch.object(app.controller, "release_all",
                                   wraps=app.controller.release_all) as rel:
                old_recognizer = app.recognizer
                created_callback = app._on_bindings_changed
                created_callback({"left_index_palm": {
                    "hand": "left", "finger": "index",
                    "action": "hold", "key": "W"}})
                rel.assert_called_once()
            self.assertIsNot(app.recognizer, old_recognizer,
                             "recognizer reloaded from saved config")
            self.assertIs(app.dispatcher.recognizer, app.recognizer)


class EmergencyStopTests(unittest.TestCase):
    def test_emergency_stop_cancels_worker_despite_release_failure(self):
        app = make_app(practice=False)
        app.engine.set_inventory_state(False)
        app.controller.set_armed(True)
        worker = mock.MagicMock()
        app.worker = worker
        app._prep_stop = threading.Event()
        generation = app._generation

        with mock.patch.object(app.controller, "release_all",
                               side_effect=RuntimeError("release boom")):
            app._emergency_stop()  # must not raise

        worker.stop.assert_called_once_with()
        self.assertIsNone(app.worker)
        self.assertGreater(app._generation, generation,
                           "generation cancelled even when release fails")
        self.assertTrue(app._prep_stop.is_set(),
                        "in-flight preparation cancelled")
        self.assertFalse(app.controller.armed)
        self.assertTrue(any("FAILED" in msg for msg in app.log_lines),
                        "cleanup error is reported, not swallowed")
        self.assertEqual(app.status_lines[-1][0], "alert")

    def test_stop_tracking_clears_all_queues(self):
        app = make_app()
        app.result_queue.put(({"Left": contact_hand()}, time.monotonic()))
        app.frame_queue.put(b"P6 1 1 255\n\x00\x00\x00")
        app.status_queue.put("stale message")
        app.stop_tracking()
        self.assertTrue(app.result_queue.empty())
        self.assertTrue(app.frame_queue.empty())
        self.assertTrue(app.status_queue.empty())

    def test_stale_generation_worker_message_ignored(self):
        app = make_app()
        app.worker = mock.MagicMock()
        app._generation = 5
        app.status_queue.put((_STATUS_WORKER_MSG, 4, "ERROR: old camera died"))
        app._tick_body()
        self.assertIsNotNone(app.worker,
                             "stale worker error must not stop this session")
        self.assertFalse(any(level == "alert"
                             for level, _ in app.status_lines))


class TickErrorTests(unittest.TestCase):
    def test_tick_error_disarms_and_cancels_before_cleanup(self):
        app = make_app(practice=True)
        app.controller.set_armed(True)
        app.sequencer.request(time.monotonic())

        armed_seen = []
        original_release = app.controller.release_all

        def spy_release(reason=""):
            armed_seen.append(app.controller.armed)
            return original_release(reason)

        app.controller.release_all = spy_release
        app.status_queue = mock.MagicMock()
        app.status_queue.get_nowait.side_effect = RuntimeError("tick boom")

        app._tick()  # must not raise

        self.assertFalse(app.controller.armed)
        self.assertFalse(app.sequencer.counting)
        self.assertTrue(armed_seen, "cleanup release ran")
        self.assertTrue(all(v is False for v in armed_seen),
                        "disarm happens BEFORE any cleanup release")
        self.assertEqual(app.status_lines[-1][0], "alert")
        app.root.after.assert_called()  # tick rescheduled even after error


class InventoryCursorSpeedTests(unittest.TestCase):
    def test_callback_pushes_var_value_to_pose_engine(self):
        app = make_app()
        pose_engine = mock.MagicMock()
        app.dispatcher.pose_engine = pose_engine
        app.inventory_cursor_speed_var = _Var(80)  # slider default

        app._on_inventory_cursor_speed("80")
        pose_engine.set_inventory_cursor_speed.assert_called_once_with(80)

        # The callback reads the IntVar at call time, not the tk string arg.
        app.inventory_cursor_speed_var.set(240)
        app._on_inventory_cursor_speed("ignored string arg")
        pose_engine.set_inventory_cursor_speed.assert_called_with(240)
        self.assertEqual(
            pose_engine.set_inventory_cursor_speed.call_count, 2)

    def test_callback_keeps_legacy_sensitivity_separate(self):
        app = make_app()
        pose_engine = mock.MagicMock()
        app.dispatcher.pose_engine = pose_engine
        app.inventory_cursor_speed_var = _Var(40)
        app.sens_var = _Var(1.5)

        app._on_inventory_cursor_speed("40")
        pose_engine.set_sensitivity.assert_not_called()

        app._on_sensitivity("1.5")
        pose_engine.set_sensitivity.assert_called_once_with(1.5)
        pose_engine.set_inventory_cursor_speed.assert_called_once_with(40)

    def test_callback_noop_without_pose_engine(self):
        # make_app's ResultDispatcher has no pose_engine attribute: the
        # slider callback must be a safe no-op, mirroring the early
        # construction path before a PoseDispatcher exists.
        app = make_app()
        app.inventory_cursor_speed_var = _Var(20)
        app._on_inventory_cursor_speed("20")  # must not raise

    def test_callback_noop_before_dispatcher_construction(self):
        # Earliest possible fire: no dispatcher or slider var at all.
        app = object.__new__(HandCraftApp)
        app._on_inventory_cursor_speed("80")  # must not raise

        # Var present but dispatcher still missing.
        app.inventory_cursor_speed_var = _Var(80)
        app._on_inventory_cursor_speed("80")  # must not raise


if __name__ == "__main__":
    unittest.main()
