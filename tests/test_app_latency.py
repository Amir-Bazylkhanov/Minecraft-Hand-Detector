"""Tick-latency tests: recognition dispatch must run before cosmetic
preview work in ``HandCraftApp._tick_body``.

Everything runs with the mocked-Tk ``make_app`` shell from
tests.test_app_safety — no camera, no real Tk window, no real OS input.
"""

import time
import tkinter as tk
import unittest
from unittest import mock

import handcraft.app as app_module
from handcraft.adapters import VK_W
from tests.test_app_safety import contact_hand, make_app


class DispatchBeforePreviewTests(unittest.TestCase):
    def test_dispatch_precedes_photoimage_and_survives_preview_error(self):
        app = make_app(practice=False, focused=True)
        app.engine.set_inventory_state(False)
        app.controller.set_armed(True)
        app.worker = mock.MagicMock()

        events = []
        real_dispatch = app.dispatcher.dispatch

        def spy_dispatch(hands, captured_at, now):
            events.append("dispatch")
            return real_dispatch(hands, captured_at, now)

        app.dispatcher.dispatch = spy_dispatch

        def boom_photoimage(*args, **kwargs):
            events.append("photoimage")
            raise RuntimeError("preview boom")

        now = time.monotonic()
        contact = {"Left": contact_hand(index_distance=0.10)}
        app.frame_queue.put(b"P6 1 1 255\n\x00\x00\x00")
        app.result_queue.put((contact, now))

        with mock.patch.object(tk, "PhotoImage", side_effect=boom_photoimage):
            app._tick_body()  # cosmetic preview error must not propagate

        self.assertEqual(events, ["dispatch", "photoimage"],
                         "recognition dispatch must run before PhotoImage")
        self.assertIn(("key_down", VK_W), app.adapter.calls,
                      "recognition happened despite the preview error")
        self.assertTrue(any("preview" in msg for msg in app.log_lines),
                        "preview failure is logged, not swallowed silently")

        # A stale packet must still release the held input even while the
        # preview keeps failing.
        app.frame_queue.put(b"P6 1 1 255\n\x00\x00\x00")
        app.result_queue.put((contact, now - 1.0))
        with mock.patch.object(tk, "PhotoImage", side_effect=boom_photoimage):
            app._tick_body()  # must not raise

        self.assertIn(("key_up", VK_W), app.adapter.calls,
                      "input release still happens despite preview errors")
        self.assertEqual(app.controller.held_keys, set())
        self.assertTrue(app.controller.armed,
                        "cosmetic preview failure must not disarm")


class StaleAgeOrderingTests(unittest.TestCase):
    def test_stale_age_uses_time_taken_after_status_before_cosmetics(self):
        app = make_app(practice=False, focused=True)
        app.engine.set_inventory_state(False)
        app.controller.set_armed(True)
        app.worker = mock.MagicMock()

        base = 5000.0
        events = []

        # Status-queue work happens first and is timestamped before `now`.
        app._preparing = True
        real_set_status = app._set_status

        def spy_status(level, text):
            events.append("status")
            return real_set_status(level, text)

        app._set_status = spy_status
        app.status_queue.put("preparing...")

        # First monotonic call is _tick_body's `now`; every later call
        # (i.e. anything during/after the expensive preview) sees a much
        # later clock.
        def fake_monotonic():
            events.append("monotonic")
            return base if events.count("monotonic") == 1 else base + 0.5

        real_dispatch = app.dispatcher.dispatch

        def spy_dispatch(hands, captured_at, now):
            events.append("dispatch")
            return real_dispatch(hands, captured_at, now)

        app.dispatcher.dispatch = spy_dispatch

        def spy_photoimage(*args, **kwargs):
            events.append("photoimage")
            return mock.MagicMock(name="photo")

        # Captured 250 ms before `now`: fresh at dispatch time (age < 300
        # ms), but stale if the age were computed AFTER the slow preview.
        captured_at = base - 0.25
        contact = {"Left": contact_hand(index_distance=0.10)}
        app.frame_queue.put(b"P6 1 1 255\n\x00\x00\x00")
        app.result_queue.put((contact, captured_at))

        with mock.patch.object(time, "monotonic", side_effect=fake_monotonic):
            with mock.patch.object(tk, "PhotoImage",
                                   side_effect=spy_photoimage):
                app._tick_body()

        self.assertLess(events.index("status"), events.index("monotonic"),
                        "status work completes before `now` is taken")
        self.assertLess(events.index("monotonic"), events.index("dispatch"),
                        "`now` is taken immediately before dispatch")
        self.assertLess(events.index("dispatch"), events.index("photoimage"),
                        "dispatch runs before preview cosmetics")
        self.assertEqual(app._last_capture_at, captured_at,
                         "packet accepted: staleness judged by the pre-"
                         "cosmetics timestamp, not by post-preview time")
        self.assertIn(("key_down", VK_W), app.adapter.calls)


class ServiceBeforeBodyTests(unittest.TestCase):
    def test_service_inputs_runs_before_body_and_photoimage(self):
        app = make_app(practice=False, focused=True)
        app.engine.set_inventory_state(False)
        app.controller.set_armed(True)
        app.worker = mock.MagicMock()

        events = []
        real_service = app._service_inputs

        def spy_service():
            events.append("service")
            return real_service()

        app._service_inputs = spy_service
        real_body = app._tick_body

        def spy_body():
            events.append("body")
            return real_body()

        app._tick_body = spy_body

        def spy_photoimage(*args, **kwargs):
            events.append("photoimage")
            return mock.MagicMock(name="photo")

        contact = {"Left": contact_hand(index_distance=0.10)}
        app.frame_queue.put(b"P6 1 1 255\n\x00\x00\x00")
        app.result_queue.put((contact, time.monotonic()))

        with mock.patch.object(tk, "PhotoImage", side_effect=spy_photoimage):
            app._tick()

        self.assertEqual(events, ["service", "body", "photoimage"],
                         "input servicing runs before the tick body, and "
                         "both run before preview cosmetics")
        app.root.after.assert_called_once_with(app_module.TICK_MS, app._tick)


class TickRescheduleTests(unittest.TestCase):
    def test_tick_interval_is_8ms(self):
        self.assertEqual(app_module.TICK_MS, 8)

    def test_tick_reschedules_after_body_error(self):
        app = make_app(practice=True)
        app.status_queue = mock.MagicMock()
        app.status_queue.get_nowait.side_effect = RuntimeError("tick boom")

        app._tick()  # must not raise

        app.root.after.assert_called_once_with(app_module.TICK_MS, app._tick)

    def test_tick_reschedules_after_service_inputs_error(self):
        app = make_app(practice=True)
        app.controller.service_inputs = mock.MagicMock(
            side_effect=RuntimeError("service boom"))

        app._tick()  # must not raise

        app.root.after.assert_called_once_with(app_module.TICK_MS, app._tick)
        self.assertEqual(app.status_lines[-1][0], "alert")


if __name__ == "__main__":
    unittest.main()
