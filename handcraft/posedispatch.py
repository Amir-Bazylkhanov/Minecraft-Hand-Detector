"""Full-pose control pipeline, sharing the existing input safety gates."""
from .app import ResultDispatcher
from .bindings import RecognitionResult
from .gestures import GestureKind
from .posemap import PoseMapEngine


class PoseDispatcher(ResultDispatcher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pose_engine = PoseMapEngine()
        self.last_pose = None

    def dispatch(self, hands, captured_at, now):
        if now - captured_at > self.max_age_s:
            return False
        hands = hands or {}
        if self._gated():
            self.tracking_lost(now)
            return True
        live_armed = self.controller.armed and not self.controller.practice
        try:
            pose = self.pose_engine.update(hands, captured_at,
                                           self.engine.inventory_believed is True)
            self.last_pose = pose
            bindings = RecognitionResult(frozenset(pose.held_keys), tuple(pose.pulse_keys),
                                         {}, {}, {}, frozenset(hands))
            self.last_result = bindings
            # A menu-toggle frame must release bindings and cancel any
            # adapter-owned timed world click BEFORE the E tap is handled:
            # a pending jab click (LMB held for ~100 ms by the adapter)
            # left down across the E press would open the inventory with
            # the attack button still held. Only the toggle frame cancels;
            # the pose engine already suppresses world keys in menus, so
            # an open menu must not cancel a valid inventory click early.
            menu_toggle = any(event.kind in (GestureKind.OPEN_INVENTORY,
                                             GestureKind.CLOSE_INVENTORY)
                              for event in pose.events)
            self.controller.apply_bindings(bindings,
                                           movement_enabled=not menu_toggle)
            for event in pose.events:
                self.controller.handle([event])
                if event.kind is GestureKind.OPEN_INVENTORY:
                    self.engine.set_inventory_state(True)
                elif event.kind is GestureKind.CLOSE_INVENTORY:
                    self.engine.set_inventory_state(False)
            self.controller.apply_pointer(pose)
        finally:
            if live_armed and not self.controller.armed:
                self.reset()
        return True

    def tracking_lost(self, now):
        self.controller.handle(self.pose_engine.reset())
        self.last_pose = None
        self.last_result = RecognitionResult(frozenset(), (), {}, {}, {}, frozenset())
        # Binding cleanup disarms on a failed release; total best-effort
        # release alone deliberately swallows errors and is not enough here.
        self.controller.apply_bindings(self.last_result, movement_enabled=False)
        self.controller.release_all("pose tracking lost")

    def reset(self, preserve_inventory=False):
        self.controller.handle(self.pose_engine.reset())
        self.controller.release_all("pose reset")
        self.last_pose = None
        super().reset(preserve_inventory)
