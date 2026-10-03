"""Integration tests: arming, stale captures, one-shot jump, W release,
preparation cancellation, practice-mode no-input.

Everything runs with mock adapter/focus/worker — no camera is opened, no
model is downloaded, and no real OS input is ever sent.
"""

import os
import queue
import tempfile
import threading
import unittest

from handcraft import tracker
from handcraft.adapters import VK_SPACE, VK_W, RecordingAdapter
from handcraft.app import (
    _STATUS_PREP_CANCELLED,
    _STATUS_WORKER_READY,
    HandCraftApp,
    ResultDispatcher,
)
from handcraft.bindings import PalmContactRecognizer
from handcraft.controller import ArmSequencer, SafetyController
from handcraft.gestures import GestureEngine
from handcraft.tracker import DownloadCancelled, ensure_model


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


def make_stack(focus=True):
    """engine + recognizer + controller + dispatcher over a recording
    adapter and a settable focus callable."""
    adapter = RecordingAdapter()
    focus_state = {"focused": focus}
    controller = SafetyController(
        adapter, focus_check=lambda: focus_state["focused"])
    controller.set_practice(False)
    engine = GestureEngine()
    recognizer = PalmContactRecognizer()
    dispatcher = ResultDispatcher(engine, recognizer, controller)
    return adapter, focus_state, controller, engine, recognizer, dispatcher


class ArmingTests(unittest.TestCase):
    def test_countdown_then_armed_once(self):
        seq = ArmSequencer(countdown_s=3.0)
        seq.request(now=10.0)
        self.assertTrue(seq.counting)
        self.assertFalse(seq.poll(12.9), "countdown not elapsed yet")
        self.assertTrue(seq.counting)
        self.assertTrue(seq.poll(13.0), "countdown elapsed -> armed")
        self.assertTrue(seq.armed)
        self.assertFalse(seq.poll(14.0), "completion fires exactly once")

    def test_cancel_stops_countdown(self):
        seq = ArmSequencer(countdown_s=3.0)
        seq.request(now=0.0)
        seq.cancel()
        self.assertFalse(seq.poll(5.0))
        self.assertFalse(seq.armed)

    def test_output_gated_on_immediate_focus_check(self):
        adapter, focus, controller, *_ = make_stack(focus=False)
        controller.set_armed(True)
        # Not focused at dispatch time: nothing is injected, and the
        # controller fails closed (disarmed).
        result = PalmContactRecognizer().update_hands(
            {"Left": contact_hand(thumb_distance=0.10)})
        controller.apply_bindings(result)
        self.assertEqual(adapter.calls, [])
        self.assertFalse(controller.armed, "focus failure must disarm")

    def test_no_stale_focus_cache_and_rearm_required(self):
        adapter, focus, controller, engine, recognizer, dispatcher = make_stack()
        controller.set_armed(True)

        jump = {"Left": contact_hand(index_distance=1.3, thumb_distance=0.10)}
        dispatcher.dispatch(jump, captured_at=100.0, now=100.0)
        self.assertIn(("tap_key", VK_SPACE), adapter.calls)

        # Focus flips between dispatches: the very next input is blocked
        # immediately (no 250 ms cache), everything is released, disarmed.
        focus["focused"] = False
        dispatcher.dispatch({"Left": contact_hand(1.3, 0.60)},
                            captured_at=100.1, now=100.1)
        dispatcher.dispatch(jump, captured_at=100.2, now=100.2)
        taps = [c for c in adapter.calls if c[0] == "tap_key"]
        self.assertEqual(len(taps), 1, "no input may pass while unfocused")
        self.assertFalse(controller.armed)

        # Focus returns: still no input until an explicit re-arm.
        focus["focused"] = True
        dispatcher.dispatch({"Left": contact_hand(1.3, 0.60)},
                            captured_at=100.3, now=100.3)
        dispatcher.dispatch(jump, captured_at=100.4, now=100.4)
        taps = [c for c in adapter.calls if c[0] == "tap_key"]
        self.assertEqual(len(taps), 1, "explicit re-arm required")

        controller.set_armed(True)
        dispatcher.dispatch({"Left": contact_hand(1.3, 0.60)},
                            captured_at=100.5, now=100.5)
        dispatcher.dispatch(jump, captured_at=100.6, now=100.6)
        taps = [c for c in adapter.calls if c[0] == "tap_key"]
        self.assertEqual(len(taps), 2)


class StaleCaptureTests(unittest.TestCase):
    def test_old_packet_discarded_before_any_action(self):
        adapter, focus, controller, engine, recognizer, dispatcher = make_stack()
        controller.set_armed(True)
        hands = {"Left": contact_hand(index_distance=0.10, thumb_distance=1.3)}
        # Captured 400 ms ago: discarded before engine/recognizer/controller.
        fresh = dispatcher.dispatch(hands, captured_at=50.0, now=50.4)
        self.assertFalse(fresh)
        self.assertEqual(adapter.calls, [])
        self.assertIsNone(dispatcher.last_result)

    def test_fresh_packet_acts(self):
        adapter, focus, controller, engine, recognizer, dispatcher = make_stack()
        controller.set_armed(True)
        hands = {"Left": contact_hand(index_distance=0.10, thumb_distance=1.3)}
        self.assertTrue(dispatcher.dispatch(hands, captured_at=50.0, now=50.1))
        self.assertIn(("key_down", VK_W), adapter.calls)


class JumpAndWalkTests(unittest.TestCase):
    def test_one_shot_jump(self):
        adapter, focus, controller, engine, recognizer, dispatcher = make_stack()
        controller.set_armed(True)
        contact = {"Left": contact_hand(index_distance=1.3, thumb_distance=0.10)}
        away = {"Left": contact_hand(index_distance=1.3, thumb_distance=0.60)}

        dispatcher.dispatch(contact, 0.0, 0.0)
        dispatcher.dispatch(contact, 0.05, 0.05)  # still touching: no repeat
        dispatcher.dispatch(contact, 0.10, 0.10)
        taps = [c for c in adapter.calls if c == ("tap_key", VK_SPACE)]
        self.assertEqual(len(taps), 1, "pulse fires once per contact")

        dispatcher.dispatch(away, 0.15, 0.15)
        dispatcher.dispatch(contact, 0.20, 0.20)  # new contact: pulse again
        taps = [c for c in adapter.calls if c == ("tap_key", VK_SPACE)]
        self.assertEqual(len(taps), 2)

    def test_w_hold_and_release(self):
        adapter, focus, controller, engine, recognizer, dispatcher = make_stack()
        controller.set_armed(True)
        contact = {"Left": contact_hand(index_distance=0.10, thumb_distance=1.3)}
        away = {"Left": contact_hand(index_distance=1.30, thumb_distance=1.3)}

        dispatcher.dispatch(contact, 0.0, 0.0)
        dispatcher.dispatch(contact, 0.05, 0.05)  # held: no duplicate key_down
        downs = [c for c in adapter.calls if c == ("key_down", VK_W)]
        self.assertEqual(len(downs), 1)
        self.assertNotIn(("key_up", VK_W), adapter.calls)

        dispatcher.dispatch(away, 0.10, 0.10)
        self.assertIn(("key_up", VK_W), adapter.calls)

    def test_w_released_on_tracking_loss(self):
        adapter, focus, controller, engine, recognizer, dispatcher = make_stack()
        controller.set_armed(True)
        dispatcher.dispatch(
            {"Left": contact_hand(index_distance=0.10, thumb_distance=1.3)},
            0.0, 0.0)
        self.assertIn(("key_down", VK_W), adapter.calls)
        dispatcher.tracking_lost(now=0.4)
        self.assertIn(("key_up", VK_W), adapter.calls)
        self.assertEqual(controller.held_keys, set())

    def test_movement_disabled_while_inventory_open(self):
        adapter, focus, controller, engine, recognizer, dispatcher = make_stack()
        controller.set_armed(True)
        engine.set_inventory_state(True)
        dispatcher.dispatch(
            {"Left": contact_hand(index_distance=0.10, thumb_distance=1.3)},
            0.0, 0.0)
        self.assertNotIn(("key_down", VK_W), adapter.calls,
                         "movement holds suppressed with inventory open")

        # A held W is released the moment the inventory opens.
        engine.set_inventory_state(False)
        contact = {"Left": contact_hand(index_distance=0.10, thumb_distance=1.3)}
        dispatcher.dispatch(contact, 0.1, 0.1)
        self.assertIn(("key_down", VK_W), adapter.calls)
        engine.set_inventory_state(True)
        dispatcher.dispatch(contact, 0.2, 0.2)
        self.assertIn(("key_up", VK_W), adapter.calls)


class PreparationCancellationTests(unittest.TestCase):
    def test_download_cancelled_mid_stream(self):
        stop = threading.Event()
        chunks = [b"x" * 4096] * 4

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, n=-1):
                if not chunks:
                    return b""
                stop.set()  # user pressed Stop/F8 mid-download
                return chunks.pop(0)

        original = tracker.urllib.request.urlopen
        tracker.urllib.request.urlopen = lambda url, timeout=None: FakeResponse()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(DownloadCancelled):
                    ensure_model(tmp, stop_event=stop, timeout=1.0)
                self.assertEqual(os.listdir(tmp), [],
                                 "no model or .part file may survive cancel")
        finally:
            tracker.urllib.request.urlopen = original

    def test_download_cancelled_before_start(self):
        stop = threading.Event()
        stop.set()

        def boom(url, timeout=None):
            raise AssertionError("network must not be touched once cancelled")

        original = tracker.urllib.request.urlopen
        tracker.urllib.request.urlopen = boom
        try:
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(DownloadCancelled):
                    ensure_model(tmp, stop_event=stop)
        finally:
            tracker.urllib.request.urlopen = original

    def _bare_app(self):
        """HandCraftApp shell without Tk: only the attributes the
        preparation path touches."""
        app = object.__new__(HandCraftApp)
        app.status_queue = queue.Queue()
        app._generation = 1
        app._prep_stop = threading.Event()
        app._preparing = True
        app.worker = None
        return app

    def test_cancelled_preparation_never_creates_worker(self):
        app = self._bare_app()
        app._prep_stop.set()  # Stop pressed while "downloading"
        with tempfile.TemporaryDirectory() as tmp:
            app.models_dir = tmp
            app._prepare_and_start(1, 0, app._prep_stop)
        kinds = []
        while True:
            try:
                msg = app.status_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(msg, tuple):
                kinds.append(msg[0])
        self.assertIn(_STATUS_PREP_CANCELLED, kinds)
        self.assertNotIn(_STATUS_WORKER_READY, kinds,
                         "cancelled preparation must never hand over a worker")

    def test_stale_generation_worker_never_started(self):
        app = self._bare_app()
        app._generation = 2  # Stop bumped the generation after prep began
        app._prep_stop.set()

        class FakeWorker:
            started = False

            def start(self):
                self.started = True

        worker = FakeWorker()
        app._on_worker_ready(1, worker)
        self.assertFalse(worker.started,
                         "superseded generation must never start the camera")
        self.assertIsNone(app.worker)


class PracticeModeTests(unittest.TestCase):
    def test_practice_sends_no_input(self):
        adapter = RecordingAdapter()
        controller = SafetyController(adapter, focus_check=lambda: True)
        controller.set_practice(True)
        controller.set_armed(True)
        engine = GestureEngine()
        recognizer = PalmContactRecognizer()
        dispatcher = ResultDispatcher(engine, recognizer, controller)

        contact = {"Left": contact_hand(index_distance=0.10,
                                        thumb_distance=0.10)}
        for i in range(5):
            dispatcher.dispatch(contact, i * 0.05, i * 0.05)
        self.assertEqual(adapter.calls, [],
                         "practice mode must never touch the OS")

    def test_mode_switch_resets_recognizer_and_inventory(self):
        adapter = RecordingAdapter()
        controller = SafetyController(adapter, focus_check=lambda: True)
        controller.set_practice(True)
        engine = GestureEngine()
        recognizer = PalmContactRecognizer()
        dispatcher = ResultDispatcher(engine, recognizer, controller)

        engine.set_inventory_state(True)
        contact = {"Left": contact_hand(index_distance=0.10,
                                        thumb_distance=1.3)}
        dispatcher.dispatch(contact, 0.0, 0.0)
        self.assertTrue(
            recognizer.update_hands(contact).contact_states["left_index_palm"],
            "contact is latched in practice")

        dispatcher.reset()  # what the app does on mode/arm transitions
        self.assertIsNone(engine.inventory_believed,
                          "inventory belief resets to UNKNOWN")
        self.assertEqual(recognizer.update_hands({}).held_keys, frozenset(),
                         "practice-held keys do not carry over")
        self.assertEqual(adapter.calls, [])


if __name__ == "__main__":
    unittest.main()
