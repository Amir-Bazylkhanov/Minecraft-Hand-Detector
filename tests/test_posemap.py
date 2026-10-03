import math
import unittest

from handcraft.geometry import dist, palm_size
from handcraft.posemap import PoseMapEngine
from handcraft.gestures import GestureKind
from tests.hands import make_hand, moved


def pose(fingers=(), thumb=False):
    hand = make_hand("fist", wrist=(.5, .5, 0))
    opened = make_hand("open", wrist=(.5, .5, 0))
    for finger in fingers:
        mcp = (5, 9, 13, 17)[finger]
        hand[mcp:mcp + 4] = opened[mcp:mcp + 4]
    if thumb:
        hand[1:5] = opened[1:5]
    return hand


def bent_hand(pinky=None, thumb=None):
    """Shaka variant with exact metrics: index/middle/ring stay curled
    while the pinky gets ``pinky`` finger-extension and the thumb gets
    ``thumb`` tip-to-index-MCP gap, both normalized by palm size. Values
    inside the hysteresis bands (.05-.15 pinky, .55-.65 thumb) are bends a
    whole-hand shape can only read as neutral/pinky — never as a fold."""
    hand = pose((3,), True)
    wrist = hand[0]
    palm = palm_size(hand)
    if pinky is not None:
        pip = hand[18]
        base = dist(wrist, pip)
        ux, uy = (pip[0] - wrist[0]) / base, (pip[1] - wrist[1]) / base
        radius = base + palm * pinky
        hand[20] = (wrist[0] + ux * radius, wrist[1] + uy * radius, wrist[2])
    if thumb is not None:
        mcp = hand[5]
        vx, vy = hand[4][0] - mcp[0], hand[4][1] - mcp[1]
        length = math.hypot(vx, vy)
        radius = palm * thumb
        hand[4] = (mcp[0] + vx / length * radius, mcp[1] + vy / length * radius, mcp[2])
    return hand


def thumb_hand(gap, point=False):
    """Fist (or point) with the thumb tip placed so the thumb-gap metric
    (tip-to-index-MCP distance / palm size) equals ``gap``: > .65 reads
    extended, < .55 folded, in between sits inside the hysteresis band a
    threshold watcher can only ride out."""
    hand = pose((0,), True) if point else pose((), True)
    mcp = hand[5]
    palm = palm_size(hand)
    vx, vy = hand[4][0] - mcp[0], hand[4][1] - mcp[1]
    length = math.hypot(vx, vy)
    radius = palm * gap
    hand[4] = (mcp[0] + vx / length * radius, mcp[1] + vy / length * radius, mcp[2])
    return hand


class TestPoseMap(unittest.TestCase):
    def test_inventory_open_cancels_a_pending_hotbar_fold(self):
        engine = PoseMapEngine()
        shaka = {'right': pose((3,), True)}
        folded = {'right': pose((3,), False)}
        engine.update(shaka, 0)
        engine.update(shaka, .15)
        engine.update(folded, .2)
        result = engine.update(folded, .35, inventory_open=True)
        self.assertEqual(result.scroll, 0)
        self.assertFalse(result.hotbar_active)

    def setUp(self):
        self.engine = PoseMapEngine()
        self.point = pose((0,))
        self.v = pose((0, 1))
        self.open = pose((0, 1, 2, 3), True)

    def anchor_neutral(self):
        """Start a left point session held still so the center locks here."""
        self.engine.update({"left": self.point}, 0)
        self.engine.update({"left": self.point}, .15)

    def test_joystick_diagonal_and_tracking_loss_releases(self):
        self.anchor_neutral()
        displaced = moved(self.point, dx=.1, dy=-.1)
        self.engine.update({"left": displaced}, .2)
        result = self.engine.update({"left": displaced}, .35)
        self.assertEqual(set(result.held_keys), {"W", "D"})
        self.assertEqual(self.engine.update(None, .4).held_keys, ())

    def test_stale_or_calibrated_center_ignored_on_new_session(self):
        self.engine.calibrate({"left": self.point})
        hand = moved(self.point, dx=-.2)
        self.engine.update({"left": hand}, 0)
        self.assertEqual(self.engine.update({"left": hand}, .2).held_keys, ())

    def test_default_center_is_first_pose_not_image_center(self):
        hand = moved(self.point, dx=-.2)
        self.engine.update({"left": hand}, 0)
        self.assertEqual(self.engine.update({"left": hand}, .2).held_keys, ())

    def test_initial_settling_never_moves(self):
        displaced = moved(self.point, dx=.2)
        self.engine.update({"left": displaced}, 0)
        self.assertEqual(self.engine.update({"left": displaced}, .05).held_keys, ())
        self.assertEqual(self.engine.update({"left": displaced}, .2).held_keys, ())

    def test_movement_and_hysteresis_after_anchor(self):
        self.anchor_neutral()
        self.assertEqual(self.engine.update({"left": moved(self.point, dx=.05)}, .2).held_keys, ())
        self.assertEqual(self.engine.update({"left": moved(self.point, dx=.1)}, .25).held_keys, ("D",))
        self.assertEqual(self.engine.update({"left": moved(self.point, dx=.06)}, .3).held_keys, ("D",))
        self.assertEqual(self.engine.update({"left": moved(self.point, dx=.04)}, .35).held_keys, ())

    def test_open_then_repoint_elsewhere_stays_neutral(self):
        self.anchor_neutral()
        self.assertEqual(self.engine.update({"left": moved(self.point, dx=.1)}, .2).held_keys, ("D",))
        self.engine.update({"left": self.open}, .25)
        elsewhere = moved(self.point, dx=.3)
        self.engine.update({"left": elsewhere}, .3)
        self.assertEqual(self.engine.update({"left": elsewhere}, .5).held_keys, ())

    def test_point_to_v_keeps_center_and_direction(self):
        self.anchor_neutral()
        forward = moved(self.point, dy=-.1)
        self.assertIn("W", self.engine.update({"left": forward}, .2).held_keys)
        sneak = moved(self.v, dy=-.1)
        self.assertEqual(self.engine.update({"left": sneak}, .25).held_keys, ())
        result = self.engine.update({"left": sneak}, .4)
        self.assertEqual(set(result.held_keys), {"W", "SHIFT"})

    def test_tracking_loss_recenters(self):
        self.anchor_neutral()
        self.assertEqual(self.engine.update({"left": moved(self.point, dx=.1)}, .2).held_keys, ("D",))
        self.engine.update({}, .25)
        elsewhere = moved(self.point, dx=-.3)
        self.engine.update({"left": elsewhere}, .3)
        self.assertEqual(self.engine.update({"left": elsewhere}, .5).held_keys, ())

    def test_reset_recenters(self):
        self.anchor_neutral()
        self.assertEqual(self.engine.update({"left": moved(self.point, dx=.1)}, .2).held_keys, ("D",))
        self.engine.reset()
        elsewhere = moved(self.point, dx=.3)
        self.engine.update({"left": elsewhere}, .3)
        self.assertEqual(self.engine.update({"left": elsewhere}, .5).held_keys, ())

    def test_inventory_chord_recenters_left(self):
        self.anchor_neutral()
        self.assertEqual(self.engine.update({"left": moved(self.point, dx=.1)}, .2).held_keys, ("D",))
        self.engine.update({"left": self.open, "right": self.open}, .25)
        elsewhere = moved(self.point, dy=.3)
        self.engine.update({"left": elsewhere}, .3)
        self.assertEqual(self.engine.update({"left": elsewhere}, .5).held_keys, ())

    def test_thumb_holds_space_without_pulses(self):
        self.anchor_neutral()
        forward = moved(self.point, dy=-.1)
        self.engine.update({"left": forward}, .2)
        self.engine.update({"left": forward}, .35)
        jumping = moved(pose((0,), True), dy=-.1)
        for t in (.4, .45, .5, .55):
            result = self.engine.update({"left": jumping}, t)
            self.assertIn("SPACE", result.held_keys)
            self.assertIn("W", result.held_keys)
            self.assertEqual(result.pulse_keys, ())

    def test_thumb_fold_releases_space_but_keeps_walking(self):
        self.anchor_neutral()
        forward = moved(self.point, dy=-.1)
        self.engine.update({"left": forward}, .2)
        self.engine.update({"left": forward}, .35)
        jumping = moved(pose((0,), True), dy=-.1)
        self.assertIn("SPACE", self.engine.update({"left": jumping}, .4).held_keys)
        result = self.engine.update({"left": forward}, .45)
        self.assertNotIn("SPACE", result.held_keys)
        self.assertIn("W", result.held_keys)

    def test_initial_settle_thumb_jumps_immediately_but_never_walks(self):
        jumping = moved(pose((0,), True), dx=.2)
        for t in (0, .05, .2):
            self.assertEqual(self.engine.update({"left": jumping}, t).held_keys, ("SPACE",))
        result = self.engine.update({"left": moved(jumping, dx=.1)}, .25)
        self.assertEqual(set(result.held_keys), {"SPACE", "D"})

    def test_thumb_only_shape_holds_space_immediately(self):
        thumb = pose((), True)
        for t in (0, .05, .2):
            result = self.engine.update({"left": thumb}, t)
            self.assertEqual(result.held_keys, ("SPACE",))
            self.assertEqual(result.pulse_keys, ())

    def test_moderate_left_thumb_extension_holds_space(self):
        # a .7 gap sat under the old stateless .8 line and was missed;
        # the relaxed .65 classification now recognizes it as thumb out
        self.assertEqual(self.engine.update({"left": thumb_hand(.7)}, 0).held_keys,
                         ("SPACE",))

    def test_left_space_holds_through_the_band_and_never_chatters(self):
        point = thumb_hand(.7, point=True)
        self.assertIn("SPACE", self.engine.update({"left": point}, 0).held_keys)
        # gaps inside the .55-.65 band keep the held state: no release
        self.assertIn("SPACE", self.engine.update(
            {"left": thumb_hand(.6, point=True)}, .05).held_keys)
        # only a real fold below .55 releases SPACE
        self.assertEqual(self.engine.update(
            {"left": thumb_hand(.5, point=True)}, .1).held_keys, ())
        # and from rest the band never presses SPACE either
        self.assertEqual(self.engine.update(
            {"left": thumb_hand(.6, point=True)}, .15).held_keys, ())

    def test_thumb_only_band_dip_holds_space_until_real_fold(self):
        # .70 -> .60 -> .50 on a thumb-only hand: the band dip reads as a
        # fist shape, but the held latch keeps feeding the hysteresis so
        # SPACE survives the band and releases only on the real fold
        self.assertIn("SPACE", self.engine.update({"left": thumb_hand(.7)}, 0).held_keys)
        self.assertIn("SPACE", self.engine.update({"left": thumb_hand(.6)}, .05).held_keys)
        self.assertEqual(self.engine.update({"left": thumb_hand(.5)}, .1).held_keys, ())

    def test_cold_fist_never_initiates_space(self):
        # folded and in-band fists without a prior extension never jump
        for i, gap in enumerate((.5, .6, .5, .6)):
            result = self.engine.update({"left": thumb_hand(gap)}, i * .05)
            self.assertEqual(result.held_keys, ())

    def test_thumb_only_band_latch_dies_on_loss_and_reset(self):
        jumping = thumb_hand(.7)
        self.assertIn("SPACE", self.engine.update({"left": jumping}, 0).held_keys)
        self.assertIn("SPACE", self.engine.update({"left": thumb_hand(.6)}, .05).held_keys)
        # tracking loss clears the latch; the band fist is cold afterwards
        self.assertEqual(self.engine.update({}, .1).held_keys, ())
        self.assertEqual(self.engine.update({"left": thumb_hand(.6)}, .15).held_keys, ())
        # same after a reset: no stale latch carries over
        self.assertIn("SPACE", self.engine.update({"left": jumping}, .2).held_keys)
        self.engine.reset()
        self.assertEqual(self.engine.update({"left": thumb_hand(.6)}, .25).held_keys, ())

    def test_v_thumb_jumps_immediately_and_while_sneak_walking(self):
        sneak_jump = pose((0, 1), True)
        # recognized first frame holds SPACE before any center exists
        self.assertEqual(self.engine.update({"left": sneak_jump}, 0).held_keys, ("SPACE",))
        # anchor a sneak session (thumb folded), then move with the thumb out
        self.engine.update({"left": self.v}, .05)
        self.engine.update({"left": self.v}, .2)
        forward = moved(sneak_jump, dy=-.1)
        result = self.engine.update({"left": forward}, .25)
        self.assertEqual(set(result.held_keys), {"W", "SHIFT", "SPACE"})

    def test_thumb_jump_while_centered_holds_only_space(self):
        self.anchor_neutral()
        result = self.engine.update({"left": pose((0,), True)}, .2)
        self.assertEqual(result.held_keys, ("SPACE",))

    def test_space_never_held_for_open_neutral_invalid_loss_or_menu(self):
        neutral = pose((0, 1, 2), True)  # ambiguous shape, thumb out
        jumping = pose((0,), True)
        self.assertEqual(self.engine.update({"left": self.open}, 0).held_keys, ())
        self.assertEqual(self.engine.update({"left": neutral}, .05).held_keys, ())
        self.assertEqual(self.engine.update({"left": [(0, 0, 0)] * 21}, .1).held_keys, ())
        self.assertEqual(self.engine.update(None, .15).held_keys, ())
        self.assertEqual(self.engine.update({"left": jumping}, .2, True).held_keys, ())
        result = self.engine.update({"left": self.open, "right": self.open}, .25)
        self.assertEqual(result.held_keys, ())

    def test_space_released_on_pose_loss_menu_and_reset(self):
        self.anchor_neutral()
        jumping = moved(pose((0,), True), dy=-.1)
        self.engine.update({"left": jumping}, .2)
        self.assertIn("SPACE", self.engine.update({"left": jumping}, .35).held_keys)
        # other pose (open palm) releases SPACE
        self.assertNotIn("SPACE", self.engine.update({"left": self.open}, .4).held_keys)
        # tracking loss releases SPACE
        self.engine.update({"left": jumping}, .45)
        self.engine.update({"left": jumping}, .6)
        self.assertIn("SPACE", self.engine.update({"left": jumping}, .65).held_keys)
        self.assertEqual(self.engine.update(None, .7).held_keys, ())
        # inventory chord releases SPACE
        self.engine.update({"left": jumping}, .75)
        self.engine.update({"left": jumping}, .9)
        self.assertIn("SPACE", self.engine.update({"left": jumping}, .95).held_keys)
        self.assertEqual(self.engine.update({"left": self.open, "right": self.open}, 1.0).held_keys, ())
        # reset releases held state; the next recognized frame re-holds
        # SPACE immediately, but walking waits for the new center lock
        self.engine.update({"left": jumping}, 1.05)
        self.engine.update({"left": jumping}, 1.2)
        self.assertIn("SPACE", self.engine.update({"left": jumping}, 1.25).held_keys)
        self.engine.reset()
        result = self.engine.update({"left": jumping}, 1.3)
        self.assertIn("SPACE", result.held_keys)
        self.assertNotIn("W", result.held_keys)

    def test_sprint_and_sneak_exclusive(self):
        self.anchor_neutral()
        far = moved(self.point, dy=-.25)
        self.engine.update({"left": far}, .2)
        self.assertIn("CTRL", self.engine.update({"left": far}, .35).held_keys)
        sneak = moved(self.v, dy=-.25)
        self.engine.update({"left": sneak}, .4)
        result = self.engine.update({"left": sneak}, .55)
        self.assertEqual(set(result.held_keys), {"W", "SHIFT"})

    def test_single_open_palm_never_toggles_inventory(self):
        for i in range(30):
            self.assertEqual(self.engine.update({"right": self.open}, i * .05).events, ())

    def test_both_palms_toggle_on_first_recognized_frame(self):
        hands = {"left": self.open, "right": self.open}
        result = self.engine.update(hands, 0)
        self.assertEqual([e.kind for e in result.events], [GestureKind.OPEN_INVENTORY])
        self.assertEqual(result.labels,
                         {"left": "Inventory chord", "right": "Inventory chord"})

    def test_held_or_moving_palms_never_repeat_toggle(self):
        events = []
        for i in range(30):
            hands = {"left": moved(self.open, dx=.01 * i),
                     "right": moved(self.open, dx=-.01 * i)}
            events.extend(self.engine.update(hands, i * .05).events)
        self.assertEqual([e.kind for e in events], [GestureKind.OPEN_INVENTORY])

    def test_exit_left_then_reentry_closes_on_first_frame(self):
        hands = {"left": self.open, "right": self.open}
        self.engine.update(hands, 0)
        self.assertEqual([e.kind for e in self.engine.update(hands, .05).events], [])
        # left palm leaves the chord, re-arming the latch
        self.engine.update({"right": self.open}, .1)
        result = self.engine.update(hands, .15, True)
        self.assertEqual([e.kind for e in result.events], [GestureKind.CLOSE_INVENTORY])

    def test_held_space_released_on_first_chord_frame(self):
        self.anchor_neutral()
        jumping = moved(pose((0,), True), dy=-.1)
        self.engine.update({"left": jumping}, .2)
        self.assertIn("SPACE", self.engine.update({"left": jumping}, .35).held_keys)
        result = self.engine.update({"left": self.open, "right": self.open}, .4)
        self.assertEqual(result.held_keys, ())
        self.assertEqual([e.kind for e in result.events], [GestureKind.OPEN_INVENTORY])

    def test_use_dwell_one_shot_and_inventory_v_never_clicks(self):
        self.assertIsNone(self.engine.update({"right": self.v}, 0).click)
        self.assertEqual(self.engine.update({"right": self.v}, .15).click, "right")
        self.assertIsNone(self.engine.update({"right": self.v}, .2).click)
        self.engine.update({}, .25)
        # an open menu removes the V click entirely: inventory V clicks nothing
        self.engine.update({"right": self.v}, .3, True)
        self.assertIsNone(self.engine.update({"right": self.v}, .45, True).click)

    def test_point_look_deadzone_and_inventory_cursor(self):
        self.engine.calibrate({"right": self.point})
        self.engine.update({"right": self.point}, 0)
        self.assertEqual(self.engine.update({"right": self.point}, .15).look_dx, 0)
        result = self.engine.update({"right": moved(self.point, dx=.15)}, .2)
        self.assertGreater(result.look_dx, 0)
        # opening the menu ends the look session; a fresh inventory point
        # session re-anchors silently — relative movement only after the
        # new center locks, and never an absolute cursor position
        result = self.engine.update({"right": self.point}, .25, True)
        self.assertIsNone(result.cursor)
        self.assertEqual((result.look_dx, result.look_dy), (0, 0))
        result = self.engine.update({"right": self.point}, .4, True)
        self.assertIsNone(result.cursor)
        self.assertEqual((result.look_dx, result.look_dy), (0, 0))

    def anchor_right(self):
        """Start a right point session held still so the look center locks."""
        self.engine.update({"right": self.point}, 0)
        self.engine.update({"right": self.point}, .15)

    def arm_hotbar(self):
        """Hold a stable shaka so the hotbar session is armed and ready."""
        shaka = pose((3,), True)
        self.engine.update({"right": shaka}, 0)
        self.engine.update({"right": shaka}, .15)
        return shaka

    def assert_repoint_elsewhere_neutral(self, t):
        """A fresh right point session held still never emits look."""
        elsewhere = moved(self.point, dx=.3, dy=.3)
        first = self.engine.update({"right": elsewhere}, t)
        locked = self.engine.update({"right": elsewhere}, t + .15)
        settled = self.engine.update({"right": elsewhere}, t + .25)
        for result in (first, locked, settled):
            self.assertEqual((result.look_dx, result.look_dy), (0, 0))

    def test_right_look_stale_calibrated_center_ignored_on_fresh_session(self):
        self.engine.calibrate({"right": self.point})
        hand = moved(self.point, dx=.2)
        self.engine.update({"right": hand}, 0)
        self.assertEqual(self.engine.update({"right": hand}, .15).look_dx, 0)
        self.assertEqual(self.engine.update({"right": hand}, .25).look_dx, 0)

    def test_right_look_initial_settle_and_lock_emit_nothing(self):
        hand = moved(self.point, dx=.2)
        for t in (0, .05, .15, .25):
            result = self.engine.update({"right": hand}, t)
            self.assertEqual((result.look_dx, result.look_dy), (0, 0))

    def test_right_look_turn_direction_after_lock(self):
        self.anchor_right()
        result = self.engine.update({"right": moved(self.point, dx=.1)}, .2)
        self.assertGreater(result.look_dx, 0)
        self.assertEqual(result.look_dy, 0)
        result = self.engine.update({"right": moved(self.point, dx=-.1)}, .25)
        self.assertLess(result.look_dx, 0)
        result = self.engine.update({"right": moved(self.point, dy=.1)}, .3)
        self.assertGreater(result.look_dy, 0)
        result = self.engine.update({"right": moved(self.point, dy=-.1)}, .35)
        self.assertLess(result.look_dy, 0)

    def test_right_look_neutral_jitter_stays_inside_deadzone(self):
        self.anchor_right()
        for i, d in enumerate((.02, -.02, .03, 0.)):
            result = self.engine.update({"right": moved(self.point, dx=d)}, .2 + i * .05)
            self.assertEqual((result.look_dx, result.look_dy), (0, 0))

    def test_right_look_center_persists_while_point_keeps_moving(self):
        self.anchor_right()
        displaced = moved(self.point, dx=.1)
        for i in range(10):
            result = self.engine.update({"right": displaced}, .2 + i * .05)
            self.assertGreater(result.look_dx, 0)
        # returning to the locked center stops the look exactly
        result = self.engine.update({"right": self.point}, .75)
        self.assertEqual((result.look_dx, result.look_dy), (0, 0))

    def test_right_look_exit_point_and_reentry_recenters(self):
        self.anchor_right()
        result = self.engine.update({"right": moved(self.point, dx=.1)}, .2)
        self.assertGreater(result.look_dx, 0)
        self.engine.update({"right": self.open}, .25)
        self.assert_repoint_elsewhere_neutral(.3)

    def test_right_look_recenters_after_tracking_loss(self):
        self.anchor_right()
        result = self.engine.update({"right": moved(self.point, dx=.1)}, .2)
        self.assertGreater(result.look_dx, 0)
        self.engine.update({}, .25)
        self.assert_repoint_elsewhere_neutral(.3)

    def test_right_look_recenters_after_inventory_cursor(self):
        self.anchor_right()
        result = self.engine.update({"right": moved(self.point, dx=.1)}, .2)
        self.assertGreater(result.look_dx, 0)
        elsewhere = moved(self.point, dx=.3)
        result = self.engine.update({"right": elsewhere}, .25, True)
        self.assertIsNone(result.cursor)
        self.assertEqual(result.look_dx, 0)
        # the menu closes while still pointing; the session restarts here
        self.assert_repoint_elsewhere_neutral(.3)

    def test_right_look_recenters_after_inventory_chord(self):
        self.anchor_right()
        result = self.engine.update({"right": moved(self.point, dx=.1)}, .2)
        self.assertGreater(result.look_dx, 0)
        self.engine.update({"left": self.open, "right": self.open}, .25)
        self.assert_repoint_elsewhere_neutral(.3)

    def test_right_look_recenters_after_reset(self):
        self.anchor_right()
        result = self.engine.update({"right": moved(self.point, dx=.1)}, .2)
        self.assertGreater(result.look_dx, 0)
        self.engine.reset()
        self.assert_repoint_elsewhere_neutral(.3)

    def test_right_look_recenters_after_time_gap(self):
        self.anchor_right()
        result = self.engine.update({"right": moved(self.point, dx=.1)}, .2)
        self.assertGreater(result.look_dx, 0)
        self.assert_repoint_elsewhere_neutral(2.)

    def anchor_cursor(self):
        """Open a menu and hold a right point still so the relative
        inventory cursor center locks at the neutral wrist."""
        self.engine.update({"right": self.point}, 0, True)
        self.engine.update({"right": self.point}, .15, True)

    def assert_cursor_repoint_neutral(self, t):
        """A fresh inventory point session held still never moves the
        cursor and never emits an absolute position."""
        elsewhere = moved(self.point, dx=.3, dy=.3)
        first = self.engine.update({"right": elsewhere}, t, True)
        locked = self.engine.update({"right": elsewhere}, t + .15, True)
        settled = self.engine.update({"right": elsewhere}, t + .25, True)
        for result in (first, locked, settled):
            self.assertIsNone(result.cursor)
            self.assertEqual((result.look_dx, result.look_dy), (0, 0))

    def test_inventory_cursor_neutral_entry_never_teleports(self):
        # entering the menu pointing from anywhere — even with the wrist
        # drifting during the settle — emits no cursor and no movement
        hand = moved(self.point, dx=.3, dy=-.2)
        for t in (0, .05, .15, .25):
            result = self.engine.update({"right": hand}, t, True)
            self.assertIsNone(result.cursor)
            self.assertEqual((result.look_dx, result.look_dy), (0, 0))

    def test_inventory_cursor_deadzone_fixed_speed_and_dt_cap(self):
        self.anchor_cursor()
        # inside the .2 palm deadzone (dx=.03 / palm .2 = .15): nothing
        result = self.engine.update({"right": moved(self.point, dx=.03)}, .2, True)
        self.assertEqual((result.look_dx, result.look_dy), (0, 0))
        # beyond the deadzone: fixed 80 px/s total speed * dt (default)
        result = self.engine.update({"right": moved(self.point, dx=.1)}, .25, True)
        self.assertEqual(result.look_dy, 0)
        self.assertAlmostEqual(result.look_dx, 80 * .05)
        # dt clamps at .08 around the same fixed center, which never
        # recenters mid-session
        result = self.engine.update({"right": moved(self.point, dx=.1)}, .45, True)
        self.assertAlmostEqual(result.look_dx, 80 * .08)
        # a diagonal keeps the same TOTAL speed: magnitude is 80 * dt
        result = self.engine.update({"right": moved(self.point, dx=-.1, dy=.1)}, .5, True)
        self.assertLess(result.look_dx, 0)
        self.assertGreater(result.look_dy, 0)
        self.assertAlmostEqual(math.hypot(result.look_dx, result.look_dy), 80 * .05)
        self.assertAlmostEqual(result.look_dx, -result.look_dy)
        # returning to the locked center stops the movement exactly
        result = self.engine.update({"right": self.point}, .55, True)
        self.assertEqual((result.look_dx, result.look_dy), (0, 0))

    def test_inventory_cursor_speed_fixed_near_and_far(self):
        self.anchor_cursor()
        # near (.1 -> offset .5) and far (.25 -> offset 1.25) wrists move
        # the cursor at the same total speed for the same dt
        near = self.engine.update({"right": moved(self.point, dx=.1)}, .2, True)
        self.assertAlmostEqual(near.look_dx, 80 * .05)
        self.assertEqual(near.look_dy, 0)
        far = self.engine.update({"right": moved(self.point, dx=.25)}, .25, True)
        self.assertAlmostEqual(far.look_dx, near.look_dx)
        # same on the diagonal: farther along it never increases speed
        near_diag = self.engine.update(
            {"right": moved(self.point, dx=.1, dy=-.1)}, .3, True)
        self.assertAlmostEqual(math.hypot(near_diag.look_dx, near_diag.look_dy),
                               80 * .05)
        far_diag = self.engine.update(
            {"right": moved(self.point, dx=.3, dy=-.3)}, .35, True)
        self.assertAlmostEqual(math.hypot(far_diag.look_dx, far_diag.look_dy),
                               math.hypot(near_diag.look_dx, near_diag.look_dy))

    def test_inventory_cursor_speed_defaults_to_80(self):
        self.assertAlmostEqual(self.engine.inventory_cursor_speed, 80.)

    def test_inventory_cursor_speed_configurable(self):
        self.anchor_cursor()
        self.engine.set_inventory_cursor_speed(120)
        self.assertAlmostEqual(self.engine.inventory_cursor_speed, 120.)
        result = self.engine.update({"right": moved(self.point, dx=.1)}, .2, True)
        self.assertAlmostEqual(result.look_dx, 120 * .05)
        # constant total speed still holds at the tuned rate: diagonal
        # magnitude equals single-axis speed for the same dt
        diag = self.engine.update(
            {"right": moved(self.point, dx=.1, dy=-.1)}, .25, True)
        self.assertAlmostEqual(math.hypot(diag.look_dx, diag.look_dy), 120 * .05)
        # the .2 deadzone is unchanged by the tuning
        result = self.engine.update({"right": moved(self.point, dx=.03)}, .3, True)
        self.assertEqual((result.look_dx, result.look_dy), (0, 0))

    def test_set_inventory_cursor_speed_clamps_to_20_240(self):
        self.engine.set_inventory_cursor_speed(5)
        self.assertAlmostEqual(self.engine.inventory_cursor_speed, 20.)
        self.engine.set_inventory_cursor_speed(1000)
        self.assertAlmostEqual(self.engine.inventory_cursor_speed, 240.)

    def test_set_inventory_cursor_speed_rejects_invalid_unchanged(self):
        self.engine.set_inventory_cursor_speed(100)
        for invalid in (None, "fast", float("nan"), float("inf"),
                        float("-inf"), object()):
            self.engine.set_inventory_cursor_speed(invalid)
            self.assertAlmostEqual(self.engine.inventory_cursor_speed, 100.,
                                   repr(invalid))

    def test_inventory_cursor_speed_survives_reset(self):
        self.engine.set_inventory_cursor_speed(150)
        self.engine.reset()
        self.assertAlmostEqual(self.engine.inventory_cursor_speed, 150.)
        self.anchor_cursor()
        result = self.engine.update({"right": moved(self.point, dx=.1)}, .2, True)
        self.assertAlmostEqual(result.look_dx, 150 * .05)

    def test_inventory_cursor_state_independent_from_look(self):
        # a locked gameplay look center never leaks into the menu, and a
        # locked cursor center never leaks back into gameplay
        self.anchor_right()
        self.assertGreater(
            self.engine.update({"right": moved(self.point, dx=.1)}, .2).look_dx, 0)
        self.assert_cursor_repoint_neutral(.25)
        # menu closed: the gameplay session also restarts from scratch
        self.assert_repoint_elsewhere_neutral(.6)

    def test_continuous_point_world_to_inventory_settles_fresh(self):
        # gameplay point held still locks the look center at t=.15
        self.engine.update({"right": self.point}, 0)
        self.engine.update({"right": self.point}, .15)
        # the menu opens while the SAME point continues, wrist drifting:
        # the old dwell must not lock the cursor center only frames in —
        # entry and everything within .12s emits nothing, and the drift
        # is adopted as the anchor
        drift1 = moved(self.point, dx=.2)
        drift2 = moved(self.point, dx=.25)
        for t, hand in ((.2, drift1), (.25, drift2), (.3, drift2)):
            result = self.engine.update({"right": hand}, t, True)
            self.assertIsNone(result.cursor)
            self.assertEqual((result.look_dx, result.look_dy), (0, 0))
            self.assertIsNone(self.engine._cursor_center)
        # the .12s lock frame itself emits nothing (center = latest drift)
        result = self.engine.update({"right": drift2}, .32, True)
        self.assertEqual((result.look_dx, result.look_dy), (0, 0))
        self.assertEqual(self.engine._cursor_center, drift2[0][:2])
        # only displacement AFTER the lock moves the cursor
        result = self.engine.update({"right": moved(drift2, dx=.1)}, .37, True)
        self.assertGreater(result.look_dx, 0)
        self.assertIsNone(result.cursor)

    def test_continuous_point_inventory_to_world_settles_fresh(self):
        # inventory point held still locks the cursor center at t=.15
        self.engine.update({"right": self.point}, 0, True)
        self.engine.update({"right": self.point}, .15, True)
        # the menu closes while the SAME point continues, wrist drifting:
        # the look session settles fresh too — no lock, no movement
        drift1 = moved(self.point, dx=.2)
        drift2 = moved(self.point, dx=.25)
        for t, hand in ((.2, drift1), (.25, drift2), (.3, drift2)):
            result = self.engine.update({"right": hand}, t)
            self.assertEqual((result.look_dx, result.look_dy), (0, 0))
            self.assertIsNone(self.engine._right_center)
        # the .12s lock frame itself emits nothing (center = latest drift)
        result = self.engine.update({"right": drift2}, .32)
        self.assertEqual((result.look_dx, result.look_dy), (0, 0))
        self.assertEqual(self.engine._right_center, drift2[0][:2])
        # only displacement AFTER the lock turns the view
        result = self.engine.update({"right": moved(drift2, dx=.1)}, .37)
        self.assertGreater(result.look_dx, 0)

    def test_inventory_cursor_recenters_on_exit_loss_chord_and_reset(self):
        for end in ("exit", "loss", "chord", "reset"):
            engine = PoseMapEngine()
            engine.update({"right": self.point}, 0, True)
            engine.update({"right": self.point}, .15, True)
            moved_point = moved(self.point, dx=.1)
            self.assertGreater(
                engine.update({"right": moved_point}, .2, True).look_dx, 0)
            if end == "exit":
                engine.update({"right": self.point}, .25)  # menu closed
            elif end == "loss":
                engine.update({}, .25)
            elif end == "chord":
                engine.update({"left": self.open, "right": self.open}, .25, True)
            else:
                engine.reset()
            elsewhere = moved(self.point, dx=.3)
            engine.update({"right": elsewhere}, .3, True)
            result = engine.update({"right": elsewhere}, .45, True)
            self.assertEqual((result.look_dx, result.look_dy), (0, 0), end)
            self.assertIsNone(result.cursor, end)
            result = engine.update({"right": elsewhere}, .5, True)
            self.assertEqual((result.look_dx, result.look_dy), (0, 0), end)

    def test_inventory_click_frame_never_moves_cursor(self):
        # locked session, thumb out; the fold lands with the wrist
        # displaced: the click frame emits ZERO movement — the frozen
        # cursor clicks the slot it was already on — and never an
        # absolute position
        self.engine.update({"right": thumb_hand(.9, point=True)}, 0, True)
        self.engine.update({"right": thumb_hand(.9, point=True)}, .15, True)
        result = self.engine.update(
            {"right": moved(thumb_hand(.5, point=True), dx=.1)}, .2, True)
        self.assertEqual(result.click, "left")
        self.assertEqual((result.look_dx, result.look_dy), (0, 0))
        self.assertIsNone(result.cursor)

    def test_inventory_thumb_arm_band_and_fold_pause_cursor_movement(self):
        # locked cursor session on a still thumb-folded point
        self.anchor_cursor()
        # the extension frame itself is paused: wrist motion moves nothing
        result = self.engine.update(
            {"right": moved(thumb_hand(.9, point=True), dx=.1)}, .2, True)
        self.assertEqual((result.look_dx, result.look_dy), (0, 0))
        self.assertIsNone(result.click)
        self.assertIsNone(result.cursor)
        # the whole armed/out phase stays paused while the wrist moves
        result = self.engine.update(
            {"right": moved(thumb_hand(.9, point=True), dx=-.1, dy=.1)}, .25, True)
        self.assertEqual((result.look_dx, result.look_dy), (0, 0))
        self.assertIsNone(result.click)
        # a band dip keeps the pause: no movement, no chatter
        result = self.engine.update(
            {"right": moved(thumb_hand(.6, point=True), dx=.1)}, .3, True)
        self.assertEqual((result.look_dx, result.look_dy), (0, 0))
        self.assertIsNone(result.click)
        # the completing fold clicks once on that very frame — the pause
        # never delays the click — and still emits no movement
        result = self.engine.update(
            {"right": moved(thumb_hand(.5, point=True), dy=.1)}, .35, True)
        self.assertEqual(result.click, "left")
        self.assertEqual((result.look_dx, result.look_dy), (0, 0))
        self.assertEqual(result.events, ())
        # holding the fold with the wrist moving never re-clicks or moves
        result = self.engine.update(
            {"right": moved(thumb_hand(.5, point=True), dx=-.1)}, .4, True)
        self.assertIsNone(result.click)
        self.assertEqual((result.look_dx, result.look_dy), (0, 0))

    def test_inventory_click_cooldown_holds_100ms_then_resumes_without_catchup(self):
        self.engine.update({"right": thumb_hand(.9, point=True)}, 0, True)
        self.engine.update({"right": thumb_hand(.9, point=True)}, .15, True)
        result = self.engine.update({"right": thumb_hand(.5, point=True)}, .2, True)
        self.assertEqual(result.click, "left")
        # the fold clicked at the settled center; the wrist is now held
        # far away, but the 100ms cooldown keeps the cursor frozen
        far = moved(thumb_hand(.5, point=True), dx=.2)
        for t in (.25, .29):
            result = self.engine.update({"right": far}, t, True)
            self.assertEqual((result.look_dx, result.look_dy), (0, 0))
            self.assertIsNone(result.click)
        # after the cooldown the same offset resumes at the constant
        # speed — one frame's worth of movement, no accumulated catchup
        result = self.engine.update({"right": far}, .34, True)
        self.assertAlmostEqual(result.look_dx, 80 * .05)
        self.assertEqual(result.look_dy, 0)

    def test_inventory_pause_and_click_state_clear_on_loss_nonpoint_and_mode(self):
        for clear in ("loss", "nonpoint", "mode"):
            engine = PoseMapEngine()
            engine.update({"right": thumb_hand(.9, point=True)}, 0, True)
            engine.update({"right": thumb_hand(.9, point=True)}, .15, True)
            result = engine.update({"right": thumb_hand(.5, point=True)}, .2, True)
            self.assertEqual(result.click, "left", clear)
            self.assertIsNotNone(engine._inv_cursor_pause_until, clear)
            if clear == "loss":
                engine.update({}, .25, True)
            elif clear == "nonpoint":
                engine.update({"right": self.v}, .25, True)
            else:
                engine.update({"right": thumb_hand(.5, point=True)}, .25)  # menu closed
            # the pause and the click cycle state die at the boundary
            self.assertIsNone(engine._inv_cursor_pause_until, clear)
            self.assertFalse(engine._inv_armed, clear)
            self.assertFalse(engine._inv_out, clear)
            # a fresh cycle afterwards still clicks (state was cleared,
            # not stuck), and a re-settled point moves normally — no
            # stale pause survives the boundary either
            engine.update({"right": thumb_hand(.9, point=True)}, .3, True)
            result = engine.update({"right": thumb_hand(.5, point=True)}, .35, True)
            self.assertEqual(result.click, "left", clear)
            engine.update({"right": self.point}, .4, True)
            engine.update({"right": self.point}, .55, True)
            result = engine.update({"right": moved(self.point, dx=.1)}, .6, True)
            self.assertGreater(result.look_dx, 0, clear)

    def test_hotbar_folds_require_shaka_and_rearm(self):
        shaka, previous, next_hand = pose((3,), True), pose((3,)), pose((), True)
        self.assertEqual(self.engine.update({"right": previous}, 0).scroll, 0)
        self.engine.update({"right": shaka}, .05)
        self.engine.update({"right": shaka}, .2)
        self.engine.update({"right": previous}, .25)
        self.assertEqual(self.engine.update({"right": previous}, .4).scroll, 1)
        self.assertEqual(self.engine.update({"right": previous}, .45).scroll, 0)
        self.engine.update({"right": next_hand}, .5)
        self.assertEqual(self.engine.update({"right": next_hand}, .65).scroll, 0)
        self.engine.update({"right": shaka}, .7)
        self.engine.update({"right": shaka}, .85)
        self.engine.update({"right": next_hand}, .9)
        self.assertEqual(self.engine.update({"right": next_hand}, 1.05).scroll, -1)

    def test_hotbar_exit_fist_never_mines(self):
        shaka = pose((3,), True)
        self.engine.update({"right": shaka}, 0)
        self.engine.update({"right": shaka}, .15)
        for i in range(20):
            result = self.engine.update({"right": moved(pose(), dx=i * .2)}, .2 + i * .05)
            self.assertIsNone(result.click)
            self.assertNotIn(GestureKind.PUNCH, [e.kind for e in result.events])

    def test_hotbar_intermediate_pinky_bend_keeps_session(self):
        self.arm_hotbar()
        bent = bent_hand(pinky=.11)  # a "neutral" shape, above the .10 fold line
        for i in range(4):
            result = self.engine.update({"right": bent}, .2 + i * .05)
            self.assertTrue(result.hotbar_active)
            self.assertEqual(result.scroll, 0)
            self.assertEqual((result.look_dx, result.look_dy), (0, 0))
            self.assertIsNone(result.click)
            self.assertEqual(result.events, ())
        # completing the fold afterwards still steps normally
        fold = bent_hand(pinky=-.5)
        self.engine.update({"right": fold}, .45)
        self.assertEqual(self.engine.update({"right": fold}, .6).scroll, -1)

    def test_hotbar_intermediate_thumb_bend_keeps_session(self):
        self.arm_hotbar()
        bent = bent_hand(thumb=.6)  # inside the .55-.65 hysteresis band
        for i in range(4):
            result = self.engine.update({"right": bent}, .2 + i * .05)
            self.assertTrue(result.hotbar_active)
            self.assertEqual(result.scroll, 0)
            self.assertEqual(result.events, ())
        fold = bent_hand(thumb=.5)
        self.engine.update({"right": fold}, .45)
        self.assertEqual(self.engine.update({"right": fold}, .6).scroll, 1)

    def test_hotbar_fold_dwell_survives_intermediate_bend(self):
        self.arm_hotbar()
        self.engine.update({"right": bent_hand(thumb=.5)}, .2)
        # a partial release stays "folded" through hysteresis, so the
        # confirmation dwell keeps running instead of restarting
        self.engine.update({"right": bent_hand(thumb=.6)}, .25)
        result = self.engine.update({"right": bent_hand(thumb=.6)}, .35)
        self.assertEqual(result.scroll, 1)

    def test_hotbar_thumb_threshold_jitter_never_repeats_or_rearms(self):
        self.arm_hotbar()
        folded = bent_hand(thumb=.5)
        self.engine.update({"right": folded}, .2)
        self.assertEqual(self.engine.update({"right": folded}, .35).scroll, 1)
        # jitter anywhere inside the band neither repeats the step nor
        # counts as a restore, so the session simply rides it out
        for i, gap in enumerate((.56, .64, .6, .63, .56)):
            result = self.engine.update({"right": bent_hand(thumb=gap)}, .4 + i * .05)
            self.assertEqual(result.scroll, 0)
            self.assertTrue(result.hotbar_active)
        self.assertEqual(self.engine.update({"right": folded}, .7).scroll, 0)

    def test_hotbar_pinky_threshold_jitter_never_repeats(self):
        self.arm_hotbar()
        folded = bent_hand(pinky=-.5)
        self.engine.update({"right": folded}, .2)
        self.assertEqual(self.engine.update({"right": folded}, .35).scroll, -1)
        for i, ext in enumerate((.11, .14, .09, .14, .11)):
            result = self.engine.update({"right": bent_hand(pinky=ext)}, .4 + i * .05)
            self.assertEqual(result.scroll, 0)
            self.assertTrue(result.hotbar_active)

    def test_hotbar_restore_both_stably_rearms(self):
        shaka = self.arm_hotbar()
        fold_thumb = bent_hand(thumb=.5)
        self.engine.update({"right": fold_thumb}, .2)
        self.assertEqual(self.engine.update({"right": fold_thumb}, .35).scroll, 1)
        self.assertEqual(self.engine.update({"right": fold_thumb}, .4).scroll, 0)
        # a brief restore inside the .12s stability dwell never re-arms
        self.engine.update({"right": shaka}, .45)
        self.engine.update({"right": fold_thumb}, .5)
        self.assertEqual(self.engine.update({"right": fold_thumb}, .65).scroll, 0)
        # restoring both stably re-arms; the other direction then steps
        self.engine.update({"right": shaka}, .7)
        self.engine.update({"right": shaka}, .85)
        fold_pinky = bent_hand(pinky=-.5)
        self.engine.update({"right": fold_pinky}, .9)
        self.assertEqual(self.engine.update({"right": fold_pinky}, 1.05).scroll, -1)

    def test_hotbar_both_folded_steps_nothing_and_never_mines(self):
        self.arm_hotbar()
        both = pose()  # fist: thumb and pinky folded, other three curled
        for i in range(6):
            result = self.engine.update({"right": moved(both, dx=.05 * i)}, .2 + i * .05)
            self.assertEqual(result.scroll, 0)
            self.assertTrue(result.hotbar_active)
            self.assertEqual(result.events, ())
        # the session is still live afterwards: a pinky fold steps normally
        fold_pinky = bent_hand(pinky=-.5)
        self.engine.update({"right": fold_pinky}, .5)
        self.assertEqual(self.engine.update({"right": fold_pinky}, .65).scroll, -1)

    def test_hotbar_fold_then_both_folded_cancels_pending_step(self):
        self.arm_hotbar()
        self.engine.update({"right": bent_hand(thumb=.5)}, .2)
        both = bent_hand(thumb=.5, pinky=-.5)
        for i in range(5):
            result = self.engine.update({"right": both}, .25 + i * .05)
            self.assertEqual(result.scroll, 0)
            self.assertTrue(result.hotbar_active)

    def test_hotbar_explicit_poses_exit_session(self):
        for exit_hand in (self.point, self.v, self.open, pose((0, 1, 2), True)):
            engine = PoseMapEngine()
            engine.update({"right": pose((3,), True)}, 0)
            engine.update({"right": pose((3,), True)}, .15)
            self.assertFalse(engine.update({"right": exit_hand}, .2).hotbar_active)
            fold = bent_hand(thumb=.5)
            engine.update({"right": fold}, .25)
            result = engine.update({"right": fold}, .4)
            self.assertEqual(result.scroll, 0)
            self.assertFalse(result.hotbar_active)

    def test_hotbar_hand_loss_clears_session(self):
        self.arm_hotbar()
        self.engine.update({"right": bent_hand(thumb=.5)}, .2)
        self.assertFalse(self.engine.update({}, .25).hotbar_active)
        fold = bent_hand(thumb=.5)
        self.engine.update({"right": fold}, .3)
        self.assertEqual(self.engine.update({"right": fold}, .45).scroll, 0)

    def test_hotbar_reset_clears_session(self):
        self.arm_hotbar()
        self.engine.update({"right": bent_hand(thumb=.5)}, .2)
        self.engine.reset()
        fold = bent_hand(thumb=.5)
        self.engine.update({"right": fold}, .25)
        result = self.engine.update({"right": fold}, .4)
        self.assertEqual(result.scroll, 0)
        self.assertFalse(result.hotbar_active)

    def run_cycle(self, engine, t, point=False):
        """One right-hand thumb extend->fold cycle, completed (folded) at t."""
        engine.update({"right": thumb_hand(.9, point)}, t - .05)
        return engine.update({"right": thumb_hand(.5, point)}, t)

    def test_cycle_click_and_inventory_blocks_gameplay_attack(self):
        self.engine.update({"right": thumb_hand(.9)}, 0)
        result = self.engine.update({"right": thumb_hand(.5)}, .05)
        # one completed cycle: exactly one left click, no PUNCH event leaks
        self.assertEqual(result.click, "left")
        self.assertEqual(result.events, ())
        # an open menu blocks the gameplay cycle entirely: no click, and
        # the burst does not survive into the menu either
        self.engine.update({"right": thumb_hand(.9)}, .1, True)
        result = self.engine.update({"right": thumb_hand(.5)}, .15, True)
        self.assertIsNone(result.click)
        self.assertEqual(result.events, ())

    def test_wrist_motion_never_attacks(self):
        # fast jab-like wrist translations of a fist: no velocity trigger
        # exists anymore, so nothing clicks and nothing punches
        fist = pose()
        for i in range(10):
            hand = moved(fist, dx=.2 if i % 2 else 0)
            result = self.engine.update({"right": hand}, i * .05)
            self.assertIsNone(result.click)
            self.assertEqual(result.events, ())

    def test_fold_alone_never_attacks(self):
        # a thumb that was never extended has nothing to complete
        for i in range(8):
            result = self.engine.update({"right": thumb_hand(.5)}, i * .05)
            self.assertIsNone(result.click)
            self.assertEqual(result.events, ())
        # same while pointing in gameplay
        for i in range(8, 12):
            result = self.engine.update({"right": thumb_hand(.5, point=True)}, i * .05)
            self.assertIsNone(result.click)
            self.assertEqual(result.events, ())

    def test_one_cycle_one_click_and_held_fold_never_repeats(self):
        self.engine.update({"right": thumb_hand(.9)}, 0)
        result = self.engine.update({"right": thumb_hand(.5)}, .05)
        self.assertEqual(result.click, "left")
        self.assertEqual(result.events, ())
        # holding the fold or hovering in the hysteresis band never repeats
        for i, gap in enumerate((.5, .6, .63, .5)):
            result = self.engine.update({"right": thumb_hand(gap)}, .1 + i * .05)
            self.assertIsNone(result.click)
            self.assertEqual(result.events, ())

    def test_thumb_hysteresis_band_neither_arms_nor_folds(self):
        # extension must cross .65: hovering inside the band never arms
        for i, gap in enumerate((.56, .64, .6, .63)):
            result = self.engine.update({"right": thumb_hand(gap)}, i * .05)
            self.assertIsNone(result.click)
            self.assertEqual(result.events, ())
        # a real extension arms; dips into the band never complete the cycle
        self.engine.update({"right": thumb_hand(.7)}, .25)
        for i, gap in enumerate((.6, .58, .62, .56)):
            self.assertIsNone(self.engine.update({"right": thumb_hand(gap)}, .3 + i * .05).click)
        # only the fold below .55 completes it: exactly one click
        result = self.engine.update({"right": thumb_hand(.5)}, .55)
        self.assertEqual(result.click, "left")
        self.assertEqual(result.events, ())

    def test_moderate_thumb_extension_previously_missed_now_cycles(self):
        # a .7 gap never crossed the old .8 extend line, so this cycle used
        # to be dropped entirely; the shared .65 line now completes it
        self.engine.update({"right": thumb_hand(.7)}, 0)
        result = self.engine.update({"right": thumb_hand(.5)}, .05)
        self.assertEqual(result.click, "left")
        self.assertEqual(result.events, ())

    def test_thumb_cycle_and_jump_survive_mirror_rotation_scale_translation(self):
        # mirrored selfie view, in-plane rotation, scaling and translation
        # leave the palm-normalized gap metric — and thus the modern thumb
        # behavior on both hands — unchanged
        def warp(hand):
            c, s = math.cos(.7), math.sin(.7)
            out = []
            for x, y, z in hand:
                x = -x
                out.append(((x * c - y * s) * 1.7 + .31,
                            (x * s + y * c) * 1.7 - .12, z * 1.7))
            return out
        engine = PoseMapEngine()
        engine.update({"right": warp(thumb_hand(.7))}, 0)
        result = engine.update({"right": warp(thumb_hand(.5))}, .05)
        self.assertEqual(result.click, "left")
        engine = PoseMapEngine()
        result = engine.update({"left": warp(thumb_hand(.7, point=True))}, 0)
        self.assertEqual(result.held_keys, ("SPACE",))

    def test_point_thumb_cycle_attacks_while_look_stays_centered(self):
        # a fresh point session held still locks the look center silently
        self.engine.update({"right": thumb_hand(.5, point=True)}, 0)
        self.engine.update({"right": thumb_hand(.5, point=True)}, .15)
        # extending the thumb keeps the still point looking at nothing
        result = self.engine.update({"right": thumb_hand(.9, point=True)}, .2)
        self.assertEqual((result.look_dx, result.look_dy), (0, 0))
        self.assertIsNone(result.click)
        # folding completes the cycle: one click, and still no look
        result = self.engine.update({"right": thumb_hand(.5, point=True)}, .25)
        self.assertEqual(result.click, "left")
        self.assertEqual(result.events, ())
        self.assertEqual((result.look_dx, result.look_dy), (0, 0))
        # the locked center still turns on real displacement afterwards
        result = self.engine.update({"right": moved(thumb_hand(.5, point=True), dx=.15)}, .3)
        self.assertGreater(result.look_dx, 0)

    def promote_hold(self, engine=None):
        """First cycle clicks; a second cycle folded inside the .6s burst
        window promotes to held mining (PUNCH forwarded, no 2nd click)."""
        engine = engine if engine is not None else self.engine
        result = self.run_cycle(engine, .2)
        self.assertEqual(result.click, "left")
        self.assertEqual(result.events, ())
        result = self.run_cycle(engine, .4)  # .2s later: inside the .6 window
        self.assertIsNone(result.click)
        self.assertEqual([e.kind for e in result.events], [GestureKind.PUNCH])
        return engine

    def test_cycle_burst_promotes_refreshes_and_lease_expires(self):
        engine = self.promote_hold()
        # a further completed cycle refreshes the hold: PUNCH, no click
        result = self.run_cycle(engine, .7)
        self.assertIsNone(result.click)
        self.assertEqual([e.kind for e in result.events], [GestureKind.PUNCH])
        # with no more cycles the .8s lease expires: the hold releases via
        # exactly one forwarded STOP_ATTACK
        stops = 0
        for i in range(1, 25):
            result = engine.update({"right": thumb_hand(.5)}, .7 + i * .05)
            self.assertIsNone(result.click)
            stops += [e.kind for e in result.events].count(GestureKind.STOP_ATTACK)
        self.assertEqual(stops, 1)
        # the burst is cleared: the next cycle is a first click again
        result = self.run_cycle(engine, 2.1)
        self.assertEqual(result.click, "left")
        self.assertEqual(result.events, ())

    def test_second_cycle_after_burst_window_is_another_single_click(self):
        self.assertEqual(self.run_cycle(self.engine, .2).click, "left")
        # waiting out the .6s burst window keeps the next cycle an
        # isolated click, never a hold
        for i in range(1, 15):
            self.engine.update({"right": thumb_hand(.5)}, .2 + i * .05)
        result = self.run_cycle(self.engine, 1.0)
        self.assertEqual(result.click, "left")
        self.assertEqual(result.events, ())

    def test_hold_maintained_while_thumb_extended_or_folded_mid_sequence(self):
        engine = self.promote_hold()
        # the thumb re-extends (arming the refresh): the hold persists
        for t in (.45, .5, .55):
            result = engine.update({"right": thumb_hand(.9)}, t)
            self.assertEqual(result.events, ())
            self.assertIsNone(result.click)
        # folding completes the refresh: PUNCH, never a click or a stop
        result = engine.update({"right": thumb_hand(.5)}, .6)
        self.assertEqual([e.kind for e in result.events], [GestureKind.PUNCH])
        self.assertIsNone(result.click)

    def test_promoted_hold_releases_on_other_pose(self):
        engine = self.promote_hold()
        result = engine.update({"right": self.open}, .5)
        self.assertEqual([e.kind for e in result.events], [GestureKind.STOP_ATTACK])
        self.assertIsNone(result.click)

    def test_promoted_hold_releases_on_tracking_loss_then_clicks_again(self):
        engine = self.promote_hold()
        result = engine.update({}, .5)
        self.assertEqual([e.kind for e in result.events], [GestureKind.STOP_ATTACK])
        # the burst is cleared: after reacquire the next cycle clicks again
        result = self.run_cycle(engine, .7)
        self.assertEqual(result.click, "left")
        self.assertEqual(result.events, ())

    def test_promoted_hold_releases_on_menu(self):
        engine = self.promote_hold()
        result = engine.update({"right": pose()}, .5, True)
        self.assertEqual([e.kind for e in result.events], [GestureKind.STOP_ATTACK])
        self.assertIsNone(result.click)

    def test_promoted_hold_releases_on_inventory_chord(self):
        engine = self.promote_hold()
        result = engine.update({"left": self.open, "right": self.open}, .5)
        kinds = [e.kind for e in result.events]
        self.assertIn(GestureKind.STOP_ATTACK, kinds)
        self.assertIn(GestureKind.OPEN_INVENTORY, kinds)

    def test_promoted_hold_releases_on_hotbar_reset_and_stale_gap(self):
        engine = self.promote_hold()
        result = engine.update({"right": pose((3,), True)}, .5)
        self.assertEqual([e.kind for e in result.events], [GestureKind.STOP_ATTACK])
        # reset releases a promoted hold too
        engine = self.promote_hold(PoseMapEngine())
        self.assertEqual([e.kind for e in engine.reset()], [GestureKind.STOP_ATTACK])
        # ... and so does a stale time gap inside update()
        engine = self.promote_hold(PoseMapEngine())
        result = engine.update({"right": thumb_hand(.5)}, 2.)
        self.assertEqual([e.kind for e in result.events], [GestureKind.STOP_ATTACK])
        self.assertIsNone(result.click)
        # but a click-only burst has no hold to release on reset
        engine = PoseMapEngine()
        engine.update({"right": thumb_hand(.9)}, 0)
        self.assertEqual(engine.update({"right": thumb_hand(.5)}, .05).click, "left")
        self.assertEqual(engine.reset(), [])

    def test_stale_gap_and_reset_emit_no_stop_without_hold(self):
        self.engine.update({"right": thumb_hand(.9)}, 0)
        self.assertEqual(self.engine.update({"right": thumb_hand(.5)}, .05).click, "left")
        # a stale gap resets the engine; with no promoted hold there is
        # nothing to release
        result = self.engine.update({"right": thumb_hand(.5)}, 2.)
        self.assertIsNone(result.click)
        self.assertEqual(result.events, ())
        self.assertEqual(self.engine.reset(), [])

    def test_armed_thumb_dies_on_tracking_loss(self):
        self.engine.update({"right": thumb_hand(.9)}, 0)   # extend: armed
        self.engine.update({}, .05)                         # lost
        result = self.engine.update({"right": thumb_hand(.5)}, .1)
        self.assertIsNone(result.click)                     # no stale arm
        self.assertEqual(result.events, ())

    def test_armed_thumb_dies_on_reset_and_chord(self):
        self.engine.update({"right": thumb_hand(.9)}, 0)
        self.engine.reset()
        result = self.engine.update({"right": thumb_hand(.5)}, .05)
        self.assertIsNone(result.click)
        # an arm also never survives the inventory chord
        self.engine.update({"right": thumb_hand(.9)}, .1)
        self.engine.update({"left": self.open, "right": self.open}, .15)
        result = self.engine.update({"right": thumb_hand(.5)}, .2)
        self.assertIsNone(result.click)
        self.assertEqual(result.events, ())

    def test_armed_thumb_never_crosses_the_menu_boundary(self):
        self.engine.update({"right": thumb_hand(.9)}, 0)   # gameplay armed
        # opening the menu on a folded-thumb point: the gameplay arm is
        # stale — no click — and the inventory cycle starts unarmed
        result = self.engine.update({"right": thumb_hand(.5, point=True)}, .05, True)
        self.assertIsNone(result.click)
        result = self.engine.update({"right": thumb_hand(.5, point=True)}, .1, True)
        self.assertIsNone(result.click)
        # an inventory click needs its own extend->fold cycle
        self.engine.update({"right": thumb_hand(.9, point=True)}, .15, True)
        result = self.engine.update({"right": thumb_hand(.5, point=True)}, .2, True)
        self.assertEqual(result.click, "left")
        # and an inventory arm never leaks back into gameplay
        self.engine.update({"right": thumb_hand(.9, point=True)}, .25, True)
        result = self.engine.update({"right": thumb_hand(.5)}, .3)
        self.assertIsNone(result.click)
        self.assertEqual(result.events, ())

    def test_inventory_point_thumb_cycle_clicks_once_and_v_never_does(self):
        self.engine.update({"right": thumb_hand(.5, point=True)}, 0, True)
        result = self.engine.update({"right": thumb_hand(.5, point=True)}, .15, True)
        # the cursor is relative now: a still point locks silently and no
        # absolute position is ever emitted
        self.assertIsNone(result.cursor)
        self.assertEqual((result.look_dx, result.look_dy), (0, 0))
        self.assertIsNone(result.click)
        # extend->fold on the point clicks the slot once
        self.assertIsNone(self.engine.update({"right": thumb_hand(.9, point=True)}, .2, True).click)
        result = self.engine.update({"right": thumb_hand(.5, point=True)}, .25, True)
        self.assertEqual(result.click, "left")
        self.assertEqual(result.events, ())
        # the held fold never repeats; a second cycle clicks again — menus
        # never promote to a mining hold
        self.assertIsNone(self.engine.update({"right": thumb_hand(.5, point=True)}, .3, True).click)
        self.engine.update({"right": thumb_hand(.9, point=True)}, .35, True)
        result = self.engine.update({"right": thumb_hand(.5, point=True)}, .4, True)
        self.assertEqual(result.click, "left")
        self.assertEqual(result.events, ())
        # V in the inventory never clicks anymore
        self.engine.update({"right": self.v}, .45, True)
        self.assertIsNone(self.engine.update({"right": self.v}, .6, True).click)

    def test_inventory_thumb_click_before_stability_never_emits_cursor(self):
        # first point frame extends the thumb; the fold lands before the
        # .12s pose dwell — the click still fires, at the CURRENT cursor:
        # no absolute position, no teleport, and no relative movement yet
        self.engine.update({"right": thumb_hand(.9, point=True)}, 0, True)
        result = self.engine.update({"right": thumb_hand(.5, point=True)}, .05, True)
        self.assertEqual(result.click, "left")
        self.assertIsNone(result.cursor)
        self.assertEqual((result.look_dx, result.look_dy), (0, 0))

    def test_late_fold_after_lease_expiry_releases_hold_and_clicks_fresh(self):
        engine = self.promote_hold()  # hold promoted, last cycle at .4
        for t in (.5, .7, .9, 1.1):   # stationary folded frames keep it alive
            engine.update({"right": thumb_hand(.5)}, t)
        # the thumb re-extends while the hold is still live (.79 < .8)
        result = engine.update({"right": thumb_hand(.9)}, 1.19)
        self.assertEqual(result.events, ())
        self.assertIsNone(result.click)
        # the fold lands .9 after the last cycle: the expired hold releases
        # first, then the preserved armed cycle completes as a fresh click
        result = engine.update({"right": thumb_hand(.5)}, 1.3)
        self.assertEqual([e.kind for e in result.events], [GestureKind.STOP_ATTACK])
        self.assertEqual(result.click, "left")

    def test_hotbar_thumb_cycles_never_attack_and_still_step(self):
        self.arm_hotbar()
        # inside the session a thumb fold only steps the hotbar slot
        self.engine.update({"right": bent_hand(thumb=.5)}, .2)
        self.assertEqual(self.engine.update({"right": bent_hand(thumb=.5)}, .35).scroll, 1)
        for i in range(4):
            result = self.engine.update({"right": bent_hand(thumb=.5)}, .4 + i * .05)
            self.assertIsNone(result.click)
            self.assertEqual(result.events, ())
        # a both-folded fist rest and a later thumb extension inside the
        # session never attack either
        fist = pose()
        for i in range(4):
            result = self.engine.update({"right": fist}, .65 + i * .05)
            self.assertIsNone(result.click)
            self.assertEqual(result.events, ())
            self.assertTrue(result.hotbar_active)
        result = self.engine.update({"right": bent_hand(thumb=.9)}, .85)
        self.assertIsNone(result.click)
        self.assertEqual(result.events, ())

    def test_stale_gap_resets_one_shots_and_emits_no_movement(self):
        self.engine.update({"left": self.point}, 0)
        self.engine.update({"left": self.point}, .15)
        result = self.engine.update({"left": moved(self.point, dx=.2)}, 2.)
        self.assertEqual(result.held_keys, ())

    def test_degenerate_or_nan_hand_does_nothing(self):
        for hand in ([ (0, 0, 0) ] * 21, [(float("nan"), 0, 0)] * 21):
            result = self.engine.update({"left": hand, "right": hand}, 0)
            self.assertEqual(result.held_keys, ())
            self.assertEqual(result.events, ())


if __name__ == "__main__":
    unittest.main()
