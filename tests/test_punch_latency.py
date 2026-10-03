"""Fast out-and-back fist movement should start on its first fast frame."""
import unittest

from handcraft.gestures import GestureEngine, GestureKind
from tests.hands import make_hand, moved


class PunchLatencyTests(unittest.TestCase):
    def test_first_fast_segment_starts_attack_at_common_frame_rates(self):
        for fps in (15, 30, 60):
            with self.subTest(fps=fps):
                engine = GestureEngine()
                fist = make_hand('fist')
                dt = 1 / fps
                for n in range(fps):
                    engine.update(fist, n * dt)
                # Six palm lengths/s: above threshold on this frame,
                # but averaging with stationary history would miss it.
                events = engine.update(moved(fist, dx=6 * .2 * dt), 1.0)
                self.assertIn(GestureKind.PUNCH, [e.kind for e in events])
                self.assertTrue(engine.attacking)
                # Return must not cancel the active mining lease.
                engine.update(fist, 1.0 + dt)
                self.assertTrue(engine.attacking)

    def test_small_stationary_jitter_does_not_attack(self):
        engine = GestureEngine()
        fist = make_hand('fist')
        events = []
        for n in range(90):
            events.extend(engine.update(moved(fist, dx=.001 * (-1) ** n), n / 60))
        self.assertNotIn(GestureKind.PUNCH, [e.kind for e in events])

    def test_long_gap_does_not_create_a_velocity_spike(self):
        engine = GestureEngine()
        fist = make_hand('fist')
        engine.update(fist, 0)
        engine.update(fist, .4)
        events = engine.update(moved(fist, dx=2), .8)
        self.assertNotIn(GestureKind.PUNCH, [e.kind for e in events])

    def test_pause_rearms_and_next_fast_frame_refreshes_lease(self):
        engine = GestureEngine()
        fist = make_hand('fist')
        for n in range(30):
            engine.update(fist, n / 30)
        hit = moved(fist, dx=.04)
        self.assertIn(GestureKind.PUNCH, [e.kind for e in engine.update(hit, 1.)])
        engine.update(hit, 1.04)
        next_hit = moved(hit, dx=.06)
        events = engine.update(next_hit, 1.2)
        # The long gap is deliberately ignored; a fresh segment then hits.
        self.assertNotIn(GestureKind.PUNCH, [e.kind for e in events])
        events = engine.update(moved(next_hit, dx=.06), 1.24)
        self.assertIn(GestureKind.PUNCH, [e.kind for e in events])


if __name__ == '__main__':
    unittest.main()
