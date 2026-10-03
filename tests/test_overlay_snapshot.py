"""Tests for HandCraftApp._update_control_overlay: the immutable control
snapshot handed from the Tk tick to the tracker worker."""
import unittest
from types import SimpleNamespace

from handcraft.app import HandCraftApp


def _bare_app(worker=None, pose_engine=None, last_pose=None, believed=False):
    """Bare app shell: no Tk, just the attributes the overlay update reads.
    Mirrors the bare-shell pattern of the other app tests; a dispatcher
    without pose_engine exercises the no-op path."""
    app = object.__new__(HandCraftApp)
    app.worker = worker
    app.dispatcher = SimpleNamespace(
        pose_engine=pose_engine, last_pose=last_pose)
    app.engine = SimpleNamespace(inventory_believed=believed)
    return app


class ControlOverlaySnapshotTest(unittest.TestCase):

    def test_worker_none_is_noop(self):
        app = _bare_app(worker=None, pose_engine=SimpleNamespace())
        app._update_control_overlay()  # must not raise

    def test_no_pose_engine_is_noop(self):
        worker = SimpleNamespace(control_overlay_state={})
        app = _bare_app(worker=worker, pose_engine=None)
        app._update_control_overlay()
        self.assertEqual(worker.control_overlay_state, {})

    def test_snapshot_data(self):
        pe = SimpleNamespace(
            _left_center=(0.1, 0.2),
            _left_anchor=(0.3, 0.4),
            _right_center=(0.5, 0.6),
            _right_anchor=(0.7, 0.8),
            centers={"right": (0.9, 0.9)},  # stale calibrated center: ignored
        )
        pose = SimpleNamespace(
            held_keys=("w",),
            labels={"left": "Move", "right": "Mine"},
            hotbar_active=True,
        )
        worker = SimpleNamespace(control_overlay_state={})
        app = _bare_app(worker=worker, pose_engine=pe, last_pose=pose,
                        believed=True)
        app._update_control_overlay()
        snap = worker.control_overlay_state
        self.assertEqual(snap["left_center"], (0.1, 0.2))
        self.assertEqual(snap["right_center"], (0.5, 0.6))
        self.assertEqual(snap["held_keys"], ("w",))
        self.assertEqual(snap["labels"], {"left": "Move", "right": "Mine"})
        self.assertIs(snap["inventory_open"], True)
        self.assertIs(snap["hotbar_active"], True)
        self.assertIs(snap["left_ready"], True)
        self.assertIs(snap["right_ready"], True)

    def test_left_anchor_fallback_and_not_ready(self):
        pe = SimpleNamespace(
            _left_center=None,
            _left_anchor=(0.3, 0.4),
            _right_center=None,
            _right_anchor=None,
            centers={},
        )
        worker = SimpleNamespace(control_overlay_state={})
        app = _bare_app(worker=worker, pose_engine=pe, last_pose=None,
                        believed=False)
        app._update_control_overlay()
        snap = worker.control_overlay_state
        self.assertEqual(snap["left_center"], (0.3, 0.4))
        self.assertIs(snap["left_ready"], False)
        self.assertIsNone(snap["right_center"])
        self.assertIs(snap["right_ready"], False)
        self.assertEqual(snap["held_keys"], ())
        self.assertEqual(snap["labels"], {})
        self.assertIs(snap["inventory_open"], False)
        self.assertIs(snap["hotbar_active"], False)

    def test_labels_and_held_keys_are_copied(self):
        pe = SimpleNamespace(_left_center=None, _left_anchor=None,
                             centers={})
        pose = SimpleNamespace(
            held_keys=["w"],
            labels={"left": "Move"},
            hotbar_active=False,
        )
        worker = SimpleNamespace(control_overlay_state={})
        app = _bare_app(worker=worker, pose_engine=pe, last_pose=pose)
        app._update_control_overlay()
        snap = worker.control_overlay_state
        self.assertIsNot(snap["labels"], pose.labels)
        pose.labels["left"] = "Mutated"
        pose.held_keys.append("space")
        self.assertEqual(snap["labels"], {"left": "Move"})
        self.assertEqual(snap["held_keys"], ("w",))

    def test_right_locked_preferred_over_stale_calibrated_center(self):
        """A locked private right center wins; the old calibrated
        centers["right"] must never drive the look overlay anymore."""
        pe = SimpleNamespace(
            _left_center=None, _left_anchor=None,
            _right_center=(0.11, 0.22),
            _right_anchor=(0.33, 0.44),
            centers={"right": (0.99, 0.99)},
        )
        worker = SimpleNamespace(control_overlay_state={})
        app = _bare_app(worker=worker, pose_engine=pe)
        app._update_control_overlay()
        snap = worker.control_overlay_state
        self.assertEqual(snap["right_center"], (0.11, 0.22))
        self.assertIs(snap["right_ready"], True)

    def test_right_anchor_fallback_and_not_ready(self):
        """Settling: no locked center yet -> private anchor is shown, but
        the right side reports not-ready even if a stale calibrated
        center exists."""
        pe = SimpleNamespace(
            _left_center=None, _left_anchor=None,
            _right_center=None,
            _right_anchor=(0.33, 0.44),
            centers={"right": (0.99, 0.99)},
        )
        worker = SimpleNamespace(control_overlay_state={})
        app = _bare_app(worker=worker, pose_engine=pe)
        app._update_control_overlay()
        snap = worker.control_overlay_state
        self.assertEqual(snap["right_center"], (0.33, 0.44))
        self.assertIs(snap["right_ready"], False)

    def test_right_clearing_none(self):
        """Neither locked nor anchor: the snapshot clears to None and the
        stale calibrated center stays out of the overlay."""
        pe = SimpleNamespace(
            _left_center=None, _left_anchor=None,
            _right_center=None,
            _right_anchor=None,
            centers={"right": (0.99, 0.99)},
        )
        worker = SimpleNamespace(control_overlay_state={})
        app = _bare_app(worker=worker, pose_engine=pe)
        app._update_control_overlay()
        snap = worker.control_overlay_state
        self.assertIsNone(snap["right_center"])
        self.assertIs(snap["right_ready"], False)

    def test_right_center_snapshot_isolation(self):
        """The snapshot holds a plain tuple copy: later mutation of the
        pose engine's center object cannot leak into the worker's view."""
        locked = [0.11, 0.22]
        pe = SimpleNamespace(
            _left_center=None, _left_anchor=None,
            _right_center=locked,
            _right_anchor=None,
            centers={},
        )
        worker = SimpleNamespace(control_overlay_state={})
        app = _bare_app(worker=worker, pose_engine=pe)
        app._update_control_overlay()
        snap = worker.control_overlay_state
        self.assertIsInstance(snap["right_center"], tuple)
        self.assertIsNot(snap["right_center"], locked)
        locked[0] = 0.99
        self.assertEqual(snap["right_center"], (0.11, 0.22))


if __name__ == "__main__":
    unittest.main()
