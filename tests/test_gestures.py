import unittest

from handcraft.gestures import GestureConfig, GestureEngine, GestureKind
from tests.hands import make_hand, moved


def kinds(events):
    return [e.kind for e in events]


def feed(engine, hand_fn, t0, t1, dt=0.05):
    """Feed frames from t0 (inclusive) to t1 (exclusive); collect events."""
    events = []
    t = t0
    while t < t1 - 1e-9:
        events.extend(engine.update(hand_fn(t), t))
        t += dt
    return events


def still(hand):
    return lambda t: hand


def moving(hand, per_frame, dt=0.05):
    base = [tuple(p) for p in hand]

    def fn(t):
        n = int(round(t / dt))
        return moved(base, dy=per_frame * n)
    return fn


class GestureTestBase(unittest.TestCase):
    def setUp(self):
        self.engine = GestureEngine(GestureConfig())
        self.fist = make_hand("fist")
        self.palm = make_hand("open")


class TestInventory(GestureTestBase):
    def test_open_palm_fires_open_once(self):
        events = feed(self.engine, still(self.palm), 0.0, 2.0)
        opens = [e for e in events if e.kind is GestureKind.OPEN_INVENTORY]
        self.assertEqual(len(opens), 1, "stable open palm must open once")
        self.assertGreaterEqual(opens[0].at, 0.45 - 1e-9,
                                "must respect the stability dwell")
        self.assertTrue(self.engine.inventory_believed)

    def test_fist_closes_once_and_blocks_repeat(self):
        feed(self.engine, still(self.palm), 0.0, 1.0)  # opens inventory
        events = feed(self.engine, still(self.fist), 1.0, 3.0)
        closes = [e for e in events if e.kind is GestureKind.CLOSE_INVENTORY]
        self.assertEqual(len(closes), 1)
        self.assertFalse(self.engine.inventory_believed)

    def test_fist_never_closes_when_believed_closed(self):
        # No prior open: a stable fist must NOT send E (E would toggle the
        # inventory open in game).
        events = feed(self.engine, still(self.fist), 0.0, 2.0)
        self.assertNotIn(GestureKind.CLOSE_INVENTORY, kinds(events))
        self.assertNotIn(GestureKind.OPEN_INVENTORY, kinds(events))

    def test_unknown_belief_opens_but_never_closes(self):
        self.engine.set_inventory_state(None)
        events = feed(self.engine, still(self.fist), 0.0, 2.0)
        self.assertNotIn(GestureKind.CLOSE_INVENTORY, kinds(events))
        events = feed(self.engine, still(self.palm), 2.0, 4.0)
        self.assertIn(GestureKind.OPEN_INVENTORY, kinds(events))

    def test_manual_sync_prevents_wrong_close(self):
        self.engine.set_inventory_state(False)
        events = feed(self.engine, still(self.fist), 0.0, 2.0)
        self.assertNotIn(GestureKind.CLOSE_INVENTORY, kinds(events))


class TestPunch(GestureTestBase):
    def test_fast_fist_punches_and_lease_expires(self):
        feed(self.engine, still(self.fist), 0.0, 0.5)  # acquire dwell
        events = feed(self.engine, moving(self.fist, 0.3), 0.5, 0.8, dt=0.05)
        self.assertIn(GestureKind.PUNCH, kinds(events))
        self.assertTrue(self.engine.attacking)
        # Hold still: lease (0.8 s) must expire into STOP_ATTACK.
        events = feed(self.engine, still(self.fist), 0.8, 2.0)
        self.assertIn(GestureKind.STOP_ATTACK, kinds(events))
        self.assertFalse(self.engine.attacking)

    def test_rearm_required_between_punches(self):
        feed(self.engine, still(self.fist), 0.0, 0.5)
        events = feed(self.engine, moving(self.fist, 0.3), 0.5, 1.0, dt=0.05)
        punches = [e for e in events if e.kind is GestureKind.PUNCH]
        self.assertEqual(len(punches), 1,
                         "continuous fast motion without a return must not "
                         "re-trigger punches")
        # Return stroke (slow) rearms...
        feed(self.engine, still(self.fist), 1.0, 1.2)
        # ...then a fresh jab refreshes the lease with a second PUNCH.
        events = feed(self.engine, moving(self.fist, 0.3), 1.2, 1.5, dt=0.05)
        self.assertIn(GestureKind.PUNCH, kinds(events))

    def test_no_punches_while_inventory_open(self):
        feed(self.engine, still(self.palm), 0.0, 1.0)  # inventory open
        events = feed(self.engine, moving(self.fist, 0.3), 1.0, 1.5, dt=0.05)
        self.assertNotIn(GestureKind.PUNCH, kinds(events))

    def test_post_close_suppression(self):
        feed(self.engine, still(self.palm), 0.0, 1.0)   # open
        events = feed(self.engine, still(self.fist), 1.0, 2.0)  # closes ~1.45
        closes = [e for e in events if e.kind is GestureKind.CLOSE_INVENTORY]
        self.assertEqual(len(closes), 1)
        close_t = closes[0].at
        # Fast motion right after closing: suppressed, no punch.
        events = feed(self.engine, moving(self.fist, 0.3),
                      close_t, close_t + 0.5, dt=0.05)
        self.assertNotIn(GestureKind.PUNCH, kinds(events))
        # After the suppression window a jab punches again.
        t1 = close_t + 0.8
        events = feed(self.engine, moving(self.fist, 0.3), t1, t1 + 0.3, dt=0.05)
        self.assertIn(GestureKind.PUNCH, kinds(events))


class TestTrackingLoss(GestureTestBase):
    def test_tracking_loss_releases_attack(self):
        feed(self.engine, still(self.fist), 0.0, 0.5)
        feed(self.engine, moving(self.fist, 0.3), 0.5, 0.8, dt=0.05)
        self.assertTrue(self.engine.attacking)
        events = self.engine.update(None, 0.85)
        self.assertIn(GestureKind.STOP_ATTACK, kinds(events))
        self.assertFalse(self.engine.attacking)

    def test_no_surprise_punch_on_reacquire(self):
        feed(self.engine, still(self.fist), 0.0, 0.5)
        feed(self.engine, moving(self.fist, 0.3), 0.5, 0.8, dt=0.05)
        self.engine.update(None, 0.85)  # tracking lost
        # Hand reappears already moving fast: acquire dwell must hold fire.
        events = feed(self.engine, moving(self.fist, 0.3), 0.9, 1.2, dt=0.05)
        self.assertNotIn(GestureKind.PUNCH, kinds(events))
        # After the dwell it may punch again.
        events = feed(self.engine, moving(self.fist, 0.3), 1.3, 1.6, dt=0.05)
        self.assertIn(GestureKind.PUNCH, kinds(events))

    def test_none_never_crashes_and_resets(self):
        for t in (0.0, 0.1, 0.2):
            self.assertEqual(self.engine.update(None, t), [])
        self.assertEqual(feed(self.engine, still(self.palm), 0.5, 1.5)[0].kind,
                         GestureKind.OPEN_INVENTORY)

    def test_reset_releases_hold(self):
        feed(self.engine, still(self.fist), 0.0, 0.5)
        feed(self.engine, moving(self.fist, 0.3), 0.5, 0.8, dt=0.05)
        events = self.engine.reset()
        self.assertIn(GestureKind.STOP_ATTACK, kinds(events))


if __name__ == "__main__":
    unittest.main()
