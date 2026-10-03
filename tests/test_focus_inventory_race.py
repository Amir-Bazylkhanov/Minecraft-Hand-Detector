"""Regression tests: the focus/inventory race between the 250 ms focus
cache and the actual Minecraft focus at dispatch time.

Repro shape: live-armed, inventory belief synced closed, the cached
``_focus_ok`` is True and focus was achieved — but the *immediate* focus
check already says Minecraft is gone. A stable right-hand open palm must
never tap E, the session must disarm at once, the belief must end UNKNOWN
(so the next Arm is refused until a manual sync), and an armed session
still waiting for its FIRST focus must not be treated as a focus loss.

Everything runs with a RecordingAdapter and scripted focus callables —
no Tk, no camera, no real OS input.
"""

import unittest
from types import SimpleNamespace

from handcraft.adapters import VK_E, RecordingAdapter
from handcraft.app import HandCraftApp, ResultDispatcher
from handcraft.bindings import PalmContactRecognizer
from handcraft.controller import ArmSequencer, SafetyController
from handcraft.gestures import GestureEngine
from tests.hands import make_hand


def make_live_app(focused=False):
    """Bare HandCraftApp shell: real engine/recognizer/controller/dispatcher
    wired exactly like the app, with a scripted immediate focus check —
    the same callable the app passes as ``minecraft_java_focused``."""
    focus = {"focused": focused}
    adapter = RecordingAdapter()
    controller = SafetyController(
        adapter, focus_check=lambda: focus["focused"])
    controller.set_practice(False)
    engine = GestureEngine()
    recognizer = PalmContactRecognizer()
    app = object.__new__(HandCraftApp)
    app.controller = controller
    app.engine = engine
    app.recognizer = recognizer
    app.sequencer = ArmSequencer()
    app.dispatcher = ResultDispatcher(
        engine, recognizer, controller, gate=app._gestures_live_enabled)
    app._focus_ok = focused
    app._focus_achieved = False
    app.log_lines = []
    app._log_action = app.log_lines.append
    app.status_lines = []
    app._set_status = lambda level, text: app.status_lines.append((level, text))
    return app, focus, adapter


def drive_open_palm(app, frames=30, dt=0.05, t0=100.0):
    """Feed ~1.5 s of stable right-hand open palm through the dispatcher —
    enough to pass the acquire and gesture dwells and fire OPEN_INVENTORY
    if the gate lets it through."""
    palm = {"right": make_hand("open")}
    t = t0
    for _ in range(frames):
        app.dispatcher.dispatch(palm, captured_at=t, now=t)
        t += dt


class CachedFocusRaceTests(unittest.TestCase):
    def test_cached_focus_true_actual_false_emits_no_e_and_stays_unknown(self):
        app, focus, adapter = make_live_app(focused=False)
        app.controller.set_armed(True)
        app.engine.set_inventory_state(False)  # manual sync: closed
        app._focus_ok = True                   # stale 250 ms cache
        app._focus_achieved = True

        drive_open_palm(app)

        self.assertNotIn(("tap_key", VK_E), adapter.calls,
                         "stale cache must never let an E tap through")
        self.assertFalse(app.controller.armed,
                         "actual focus loss after achieve must disarm now")
        self.assertIsNone(app.engine.inventory_believed,
                          "belief must end UNKNOWN, not a stale known value")
        self.assertFalse(app._focus_achieved)
        self.assertFalse(app._focus_ok)

    def test_focus_flips_after_gate_precheck_before_e_disarms_unknown(self):
        app, focus, adapter = make_live_app(focused=True)
        app.controller.set_armed(True)
        app.engine.set_inventory_state(False)
        app._focus_ok = True
        app._focus_achieved = True

        engine = app.engine

        def flip_after_belief_changes():
            # Focus is lost in the instant AFTER the dispatcher gate's
            # precheck passes but BEFORE the E tap is dispatched: the
            # engine has already captured the open-palm belief.
            return engine.inventory_believed is not True

        app.controller._focus_check = flip_after_belief_changes

        drive_open_palm(app)

        self.assertNotIn(("tap_key", VK_E), adapter.calls)
        self.assertFalse(app.controller.armed,
                         "focus failure at dispatch must disarm")
        self.assertIsNone(engine.inventory_believed,
                          "a belief whose E never left is not knowledge")
        self.assertFalse(engine.attacking,
                         "transient engine tracking is reset")

    def test_waiting_for_first_focus_is_not_a_focus_loss(self):
        app, focus, adapter = make_live_app(focused=False)
        app.engine.set_inventory_state(False)
        app.sequencer.request(0.0)
        self.assertTrue(app.sequencer.poll(3.0))  # countdown completed
        app.controller.set_armed(True)
        app._focus_ok = False
        app._focus_achieved = False  # game window not focused yet

        drive_open_palm(app)

        self.assertTrue(app.controller.armed,
                        "armed-waiting must survive: no disarm pre-focus")
        self.assertTrue(app.sequencer.armed,
                        "waiting state is not cancelled")
        self.assertEqual(app.engine.inventory_believed, False,
                         "belief untouched while waiting for first focus")
        self.assertNotIn(("tap_key", VK_E), adapter.calls)

    def test_next_arm_refused_until_manual_sync(self):
        app, focus, adapter = make_live_app(focused=False)
        app.practice_var = SimpleNamespace(get=lambda: False)
        app.controller.set_armed(True)
        app.engine.set_inventory_state(False)
        app._focus_ok = True
        app._focus_achieved = True

        drive_open_palm(app)
        self.assertIsNone(app.engine.inventory_believed)

        app._on_arm()  # must refuse: belief is UNKNOWN after the race
        self.assertFalse(app.sequencer.counting,
                         "no countdown while the belief is UNKNOWN")
        self.assertFalse(app.controller.armed)
        self.assertTrue(any("refused" in msg for msg in app.log_lines))

        app.engine.set_inventory_state(False)  # manual Mark closed sync
        app._on_arm()
        self.assertTrue(app.sequencer.counting,
                        "known belief after manual sync arms the countdown")

    def test_belief_captured_when_focus_actually_holds(self):
        # Positive control: state changes ARE captured when the immediate
        # focus check succeeds all the way through the dispatch.
        app, focus, adapter = make_live_app(focused=True)
        app.controller.set_armed(True)
        app.engine.set_inventory_state(False)
        app._focus_ok = True
        app._focus_achieved = True

        drive_open_palm(app)

        self.assertIn(("tap_key", VK_E), adapter.calls,
                      "focused + armed: a stable open palm taps E")
        self.assertIs(app.engine.inventory_believed, True)
        self.assertTrue(app.controller.armed)


class DispatchDisarmInvalidationTests(unittest.TestCase):
    def test_input_failure_at_dispatch_leaves_belief_unknown(self):
        # An adapter error at the E tap disarms mid-dispatch and re-raises;
        # the dispatcher must still invalidate the just-advanced belief.
        class FailingAdapter(RecordingAdapter):
            def tap_key(self, vk):
                super().tap_key(vk)
                raise RuntimeError("SendInput boom")

        adapter = FailingAdapter()
        controller = SafetyController(adapter, focus_check=lambda: True)
        controller.set_practice(False)
        controller.set_armed(True)
        engine = GestureEngine()
        engine.set_inventory_state(False)
        dispatcher = ResultDispatcher(
            engine, PalmContactRecognizer(), controller)

        palm = {"right": make_hand("open")}
        raised = None
        t = 100.0
        for _ in range(30):
            try:
                dispatcher.dispatch(palm, captured_at=t, now=t)
            except RuntimeError as exc:
                raised = exc
                break
            t += 0.05

        self.assertIsNotNone(raised, "the adapter failure must propagate")
        self.assertFalse(controller.armed, "input failure disarms")
        self.assertIsNone(engine.inventory_believed,
                          "belief invalidated even though the error raised")
        self.assertFalse(engine.attacking)


if __name__ == "__main__":
    unittest.main()
