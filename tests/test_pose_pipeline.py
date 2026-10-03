"""New full-pose map reaches the guarded keyboard/mouse adapter."""
import unittest
from unittest import mock

from handcraft.app import HandCraftApp
from handcraft.posedispatch import PoseDispatcher
from handcraft.adapters import RecordingAdapter, VK_E, VK_W, VK_SPACE
from handcraft.bindings import PalmContactRecognizer
from handcraft.controller import SafetyController
from handcraft.gestures import GestureEngine
from handcraft import win32input
from tests.test_posemap import pose, bent_hand, thumb_hand
from tests.hands import moved


class PosePipelineTests(unittest.TestCase):
    def test_inventory_minimum_gain_slow_diagonal_reaches_adapter(self):
        self.engine.set_inventory_state(True)
        self.dispatch.pose_engine.set_inventory_cursor_speed(20)
        hand = pose((0,))
        self.feed({"right": hand}, 0)
        self.feed({"right": hand}, .15)
        from handcraft.geometry import palm_size
        step = .004 * palm_size(hand)
        for i in range(1, 21):
            current = moved(hand, dx=step * i, dy=step * i)
            self.feed({"right": current}, .15 + i * .02)
        pixels = [args for name, args in self.adapter.calls if name == "move_relative"]
        self.assertAlmostEqual(sum(x for x, y in pixels), 8, delta=1)
        self.assertAlmostEqual(sum(y for x, y in pixels), 8, delta=1)
        self.adapter.calls.clear()
        for i in range(21, 41):
            self.feed({"right": current}, .15 + i * .02)
        self.assertNotIn("move_relative", self.adapter.kinds())

    def setUp(self):
        self.adapter = RecordingAdapter()
        self.focus = True
        self.ctl = SafetyController(self.adapter, focus_check=lambda: self.focus)
        self.ctl.set_practice(False)
        self.ctl.set_armed(True)
        self.engine = GestureEngine()
        self.dispatch = PoseDispatcher(self.engine, PalmContactRecognizer(), self.ctl)

    def feed(self, hands, t):
        self.dispatch.dispatch(hands, t, t)

    def hold_jump(self):
        """Walk forward, extend the thumb, and return with SPACE held."""
        point = pose((0,))
        self.feed({'left': point}, 0)
        self.feed({'left': point}, .15)
        self.feed({'left': moved(point, dy=-.1)}, .2)
        self.feed({'left': moved(point, dy=-.1)}, .35)
        self.feed({'left': moved(pose((0,), True), dy=-.1)}, .4)
        self.assertIn(VK_W, self.ctl.held_keys)
        self.assertIn(VK_SPACE, self.ctl.held_keys)

    def test_walk_jump_then_tracking_loss(self):
        self.hold_jump()
        self.assertIn(('key_down', VK_W), self.adapter.calls)
        self.assertIn(('key_down', VK_SPACE), self.adapter.calls)
        self.assertNotIn(('tap_key', VK_SPACE), self.adapter.calls)
        self.dispatch.tracking_lost(.45)
        self.assertFalse(self.ctl.held_keys)
        self.assertIn(('key_up', VK_SPACE), self.adapter.calls)

    def test_held_jump_never_repeats_key_down(self):
        self.hold_jump()
        jumping = moved(pose((0,), True), dy=-.1)
        for i in range(5):
            self.feed({'left': jumping}, .45 + i * .05)
        presses = [c for c in self.adapter.calls if c == ('key_down', VK_SPACE)]
        self.assertEqual(len(presses), 1)
        self.assertIn(VK_SPACE, self.ctl.held_keys)

    def test_thumb_fold_releases_space_preserves_walk(self):
        self.hold_jump()
        self.feed({'left': moved(pose((0,)), dy=-.1)}, .45)
        self.assertIn(('key_up', VK_SPACE), self.adapter.calls)
        self.assertNotIn(('key_up', VK_W), self.adapter.calls)
        self.assertEqual(self.ctl.held_keys, {VK_W})

    def test_thumb_only_jump_presses_space_once_without_walking(self):
        thumb = {'left': pose((), True)}
        for i in range(5):
            self.feed(thumb, i * .05)
        presses = [c for c in self.adapter.calls if c == ('key_down', VK_SPACE)]
        self.assertEqual(len(presses), 1)
        self.assertEqual(self.ctl.held_keys, {VK_SPACE})
        self.assertNotIn(('key_down', VK_W), self.adapter.calls)
        self.assertNotIn(('tap_key', VK_SPACE), self.adapter.calls)

    def test_thumb_fold_without_walking_releases_space(self):
        self.feed({'left': pose((), True)}, 0)
        self.assertEqual(self.ctl.held_keys, {VK_SPACE})
        self.feed({'left': pose()}, .05)
        self.assertFalse(self.ctl.held_keys)
        self.assertIn(('key_up', VK_SPACE), self.adapter.calls)
        self.assertNotIn(('key_up', VK_W), self.adapter.calls)

    def test_inventory_chord_releases_held_jump_on_first_frame(self):
        self.hold_jump()
        palms = {'left': pose((0, 1, 2, 3), True), 'right': pose((0, 1, 2, 3), True)}
        self.feed(palms, .45)
        self.assertTrue(self.engine.inventory_believed)
        self.assertFalse(self.ctl.held_keys)
        self.assertIn(('tap_key', VK_E), self.adapter.calls)
        up_space = self.adapter.calls.index(('key_up', VK_SPACE))
        tap_e = self.adapter.calls.index(('tap_key', VK_E))
        self.assertLess(up_space, tap_e)
        # holding the chord never repeats the E tap
        taps = [c for c in self.adapter.calls if c == ('tap_key', VK_E)]
        self.assertEqual(len(taps), 1)

    def test_chord_reentry_taps_e_once_per_entry(self):
        palms = {'left': pose((0, 1, 2, 3), True), 'right': pose((0, 1, 2, 3), True)}
        self.feed(palms, 0)
        self.feed(palms, .05)
        self.feed({'right': pose((0, 1, 2, 3), True)}, .1)
        self.feed(palms, .15)
        self.feed(palms, .2)
        taps = [c for c in self.adapter.calls if c == ('tap_key', VK_E)]
        self.assertEqual(len(taps), 2)
        self.assertFalse(self.engine.inventory_believed)

    def test_focus_loss_releases_held_jump_and_disarms(self):
        self.hold_jump()
        self.focus = False
        self.feed({'left': moved(pose((0,), True), dy=-.1)}, .45)
        self.assertFalse(self.ctl.armed)
        self.assertFalse(self.ctl.held_keys)
        self.assertIn(('key_up', VK_SPACE), self.adapter.calls)

    def test_failed_walk_release_on_tracking_loss_disarms(self):
        point = pose((0,))
        self.feed({'left': point}, 0)
        self.feed({'left': point}, .15)
        forward = moved(point, dy=-.1)
        self.feed({'left': forward}, .2)
        self.feed({'left': forward}, .35)
        with mock.patch.object(self.adapter, 'key_up', side_effect=OSError('release failed')):
            with self.assertRaises(OSError):
                self.dispatch.tracking_lost(.4)
        self.assertFalse(self.ctl.armed)
        self.assertIn(VK_W, self.ctl.held_keys)
        self.ctl.release_all('retry')
        self.assertFalse(self.ctl.held_keys)

    def test_inventory_chord_then_relative_cursor_and_thumb_cycle_slot_click(self):
        palms = {'left': pose((0, 1, 2, 3), True), 'right': pose((0, 1, 2, 3), True)}
        self.feed(palms, 0)
        self.assertTrue(self.engine.inventory_believed)
        point = {'right': thumb_hand(.5, point=True)}
        self.feed(point, .1)
        self.feed(point, .25)
        # a still point re-anchors silently: no absolute cursor, no move
        self.assertNotIn('move_cursor', self.adapter.kinds())
        self.assertNotIn('move_relative', self.adapter.kinds())
        # wrist displacement from the locked center moves relatively
        self.feed({'right': moved(point['right'], dx=.1)}, .3)
        self.assertIn('move_relative', self.adapter.kinds())
        self.assertNotIn('move_cursor', self.adapter.kinds())
        # V no longer clicks slots in the inventory
        v = {'right': pose((0, 1))}
        self.feed(v, .35)
        self.feed(v, .5)
        self.assertNotIn('click_mouse', self.adapter.kinds())
        # the point's thumb extend->fold cycle clicks the slot instead
        self.feed({'right': thumb_hand(.9, point=True)}, .55)
        self.feed({'right': thumb_hand(.5, point=True)}, .6)
        self.assertIn(('click_mouse', 'left'), self.adapter.calls)
        self.assertNotIn(('click_mouse', 'right'), self.adapter.calls)

    def test_focus_loss_pointer_dispatch_invalidates_inventory(self):
        point = {'right': pose((0,))}
        self.feed(point, 0)
        self.feed(point, .15)
        self.focus = False
        self.feed({'right': moved(point['right'], dx=.2)}, .2)
        self.assertFalse(self.ctl.armed)
        self.assertIsNone(self.engine.inventory_believed)
        self.assertNotIn('move_relative', self.adapter.kinds())

    def test_right_look_settles_before_first_turn(self):
        point = pose((0,))
        self.feed({'right': point}, 0)
        self.feed({'right': point}, .15)
        self.assertNotIn('move_relative', self.adapter.kinds())
        self.feed({'right': moved(point, dx=.15)}, .2)
        self.assertIn('move_relative', self.adapter.kinds())

    def test_right_look_stale_calibrated_center_never_turns_while_still(self):
        self.dispatch.pose_engine.calibrate({'right': pose((0,))})
        hand = moved(pose((0,)), dx=.2)
        self.feed({'right': hand}, 0)
        self.feed({'right': hand}, .15)
        self.feed({'right': hand}, .25)
        self.assertNotIn('move_relative', self.adapter.kinds())

    def test_right_look_reanchors_after_tracking_loss(self):
        point = pose((0,))
        self.feed({'right': point}, 0)
        self.feed({'right': point}, .15)
        self.feed({'right': moved(point, dx=.15)}, .2)
        self.assertIn('move_relative', self.adapter.kinds())
        self.dispatch.tracking_lost(.25)
        elsewhere = moved(point, dx=.3)
        self.feed({'right': elsewhere}, .3)
        self.feed({'right': elsewhere}, .45)
        self.feed({'right': elsewhere}, .55)
        relative = [c for c in self.adapter.calls if c[0] == 'move_relative']
        self.assertEqual(len(relative), 1)

    def test_practice_never_injects_pointer(self):
        self.ctl.set_practice(True)
        v = {'right': pose((0, 1))}
        self.feed(v, 0)
        self.feed(v, .15)
        self.assertNotIn('click_mouse', self.adapter.kinds())

    def test_hotbar_scroll_reaches_adapter_with_sign(self):
        shaka = {'right': pose((3,), True)}
        self.feed(shaka, 0)
        self.feed(shaka, .15)
        previous = {'right': pose((3,))}  # fold thumb, pinky stays out
        self.feed(previous, .2)
        self.feed(previous, .35)
        self.feed(shaka, .4)  # restore both stably to re-arm
        self.feed(shaka, .55)
        next_hand = {'right': pose((), True)}  # fold pinky, thumb stays out
        self.feed(next_hand, .6)
        self.feed(next_hand, .75)
        scrolls = [c for c in self.adapter.calls if c[0] == 'scroll']
        self.assertEqual(scrolls, [('scroll', 1), ('scroll', -1)])

    def test_hotbar_partial_folds_inject_nothing(self):
        shaka = {'right': pose((3,), True)}
        self.feed(shaka, 0)
        self.feed(shaka, .15)
        # intermediate bends and the both-folded rest: no scroll, look,
        # click or mining may leak to the OS
        for i, hand in enumerate((bent_hand(thumb=.6), bent_hand(pinky=.11),
                                  bent_hand(thumb=.6, pinky=.12), pose())):
            self.feed({'right': hand}, .2 + i * .05)
        self.assertEqual(self.adapter.calls, [])

    def test_hotbar_session_cleared_by_tracking_loss(self):
        shaka = {'right': pose((3,), True)}
        self.feed(shaka, 0)
        self.feed(shaka, .15)
        fold = {'right': pose((3,))}
        self.feed(fold, .2)
        self.dispatch.tracking_lost(.25)
        self.feed(fold, .3)
        self.feed(fold, .45)
        self.assertEqual(self.adapter.calls, [])

    def test_hotbar_pending_fold_dies_with_inventory_chord(self):
        shaka = {'right': pose((3,), True)}
        self.feed(shaka, 0)
        self.feed(shaka, .15)
        fold = {'right': pose((3,))}
        self.feed(fold, .2)
        palms = {'left': pose((0, 1, 2, 3), True), 'right': pose((0, 1, 2, 3), True)}
        self.feed(palms, .25)
        self.feed(palms, .3)
        self.feed(fold, .35)
        self.feed(fold, .5)
        self.assertIn(('tap_key', VK_E), self.adapter.calls)
        self.assertNotIn('scroll', self.adapter.kinds())

    def feed_cycle(self, t, point=False):
        """Feed one right-hand thumb extend->fold cycle completed at t."""
        self.feed({'right': thumb_hand(.9, point)}, t - .05)
        self.feed({'right': thumb_hand(.5, point)}, t)

    def test_one_cycle_one_click_reaches_adapter(self):
        self.feed_cycle(.2)
        clicks = [c for c in self.adapter.calls if c == ('click_mouse', 'left')]
        self.assertEqual(len(clicks), 1)
        # the cycle click is a timed click, never a lease-held button
        self.assertNotIn(('left_down', None), self.adapter.calls)
        # holding the fold never clicks again
        for i in range(1, 8):
            self.feed({'right': thumb_hand(.5)}, .2 + i * .05)
        clicks = [c for c in self.adapter.calls if c == ('click_mouse', 'left')]
        self.assertEqual(len(clicks), 1)
        self.assertNotIn(('left_down', None), self.adapter.calls)

    def test_moderate_thumb_extension_cycle_reaches_adapter(self):
        # a .7 gap extension (missed by the old .8 line) now clicks once
        self.feed({'right': thumb_hand(.7)}, 0)
        self.feed({'right': thumb_hand(.5)}, .05)
        clicks = [c for c in self.adapter.calls if c == ('click_mouse', 'left')]
        self.assertEqual(len(clicks), 1)
        self.assertNotIn(('left_down', None), self.adapter.calls)

    def test_wrist_motion_never_clicks(self):
        # no wrist-velocity trigger remains: jab-like fist moves do nothing
        fist = pose()
        for i in range(10):
            self.feed({'right': moved(fist, dx=.2 if i % 2 else 0)}, i * .05)
        self.assertNotIn('click_mouse', self.adapter.kinds())
        self.assertNotIn(('left_down', None), self.adapter.calls)

    def test_fold_alone_never_clicks(self):
        for i in range(10):
            self.feed({'right': thumb_hand(.5)}, i * .05)
        self.assertNotIn('click_mouse', self.adapter.kinds())
        self.assertNotIn(('left_down', None), self.adapter.calls)

    def test_cycle_while_inventory_open_never_attacks(self):
        self.engine.set_inventory_state(True)
        self.feed_cycle(.2)
        self.assertNotIn('click_mouse', self.adapter.kinds())
        self.assertNotIn(('left_down', None), self.adapter.calls)

    def promote_cycle_hold(self):
        """Feed a first cycle (one click) then a second in-burst cycle that
        promotes to a lease-held mine (one left_down, no second click)."""
        self.feed_cycle(.2)   # first cycle: click
        self.feed_cycle(.4)   # second cycle within .6s: promote to hold
        self.assertEqual(self.adapter.calls.count(('click_mouse', 'left')), 1)
        self.assertEqual(self.adapter.calls.count(('left_down', None)), 1)
        self.assertNotIn(('left_up', None), self.adapter.calls)

    def test_second_cycle_promotes_to_hold_and_refresh_never_represses(self):
        self.promote_cycle_hold()
        self.feed({'right': thumb_hand(.5)}, .45)  # held still
        self.feed_cycle(.6)                         # refresh cycle
        # the refresh only extends the lease: no new down, no new click
        self.assertEqual(self.adapter.calls.count(('click_mouse', 'left')), 1)
        self.assertEqual(self.adapter.calls.count(('left_down', None)), 1)
        self.assertTrue(self.ctl.mouse_held)
        # tracking loss releases the promoted hold exactly once
        self.dispatch.tracking_lost(.7)
        self.assertEqual(self.adapter.calls.count(('left_up', None)), 1)
        self.assertFalse(self.ctl.mouse_held)

    def test_promoted_hold_releases_on_lease_expiry(self):
        self.promote_cycle_hold()
        # no more cycles: the .8s lease expires and the forwarded
        # STOP_ATTACK releases the button exactly once
        for i in range(1, 25):
            self.feed({'right': thumb_hand(.5)}, .4 + i * .05)
        self.assertEqual(self.adapter.calls.count(('left_down', None)), 1)
        self.assertEqual(self.adapter.calls.count(('left_up', None)), 1)
        self.assertEqual(self.adapter.calls.count(('click_mouse', 'left')), 1)
        self.assertFalse(self.ctl.mouse_held)

    def test_late_fold_releases_expired_hold_before_fresh_click(self):
        self.promote_cycle_hold()          # click at .2, hold at .4
        for t in (.5, .7, .9, 1.1):
            self.feed({'right': thumb_hand(.5)}, t)
        self.feed({'right': thumb_hand(.9)}, 1.19)  # re-arm, hold still live
        self.feed({'right': thumb_hand(.5)}, 1.3)   # late fold: STOP + click
        calls = self.adapter.calls
        self.assertEqual(calls.count(('left_down', None)), 1)
        self.assertEqual(calls.count(('left_up', None)), 1)
        clicks = [i for i, c in enumerate(calls) if c == ('click_mouse', 'left')]
        self.assertEqual(len(clicks), 2)
        # the expired hold's release is ordered before the fresh click
        self.assertLess(calls.index(('left_up', None)), clicks[1])

    def test_inventory_thumb_click_never_moves_relative(self):
        self.engine.set_inventory_state(True)
        # lock the cursor session on a still point with the thumb out,
        # then fold with the wrist displaced: the click pause suppresses
        # that frame's movement entirely — the click lands at the CURRENT
        # cursor and no relative movement is dispatched at all
        point = thumb_hand(.9, point=True)
        self.feed({'right': point}, 0)
        self.feed({'right': point}, .15)
        self.feed({'right': moved(thumb_hand(.5, point=True), dx=.1)}, .2)
        self.assertIn(('click_mouse', 'left'), self.adapter.calls)
        self.assertNotIn('move_relative', self.adapter.kinds())
        self.assertNotIn('move_cursor', self.adapter.kinds())

    def test_inventory_early_thumb_click_never_moves_cursor(self):
        self.engine.set_inventory_state(True)
        # extend on the first point frame, fold before the .12s dwell:
        # the click dispatches at the CURRENT cursor — no movement at all
        self.feed({'right': thumb_hand(.9, point=True)}, 0)
        self.feed({'right': thumb_hand(.5, point=True)}, .05)
        self.assertIn(('click_mouse', 'left'), self.adapter.calls)
        self.assertNotIn('move_cursor', self.adapter.kinds())
        self.assertNotIn('move_relative', self.adapter.kinds())

    def test_inventory_relative_cursor_gated_by_practice_and_focus(self):
        self.engine.set_inventory_state(True)
        # practice mode recognizes everything but injects nothing
        self.ctl.set_practice(True)
        point = {'right': pose((0,))}
        self.feed(point, 0)
        self.feed(point, .15)
        self.feed({'right': moved(point['right'], dx=.15)}, .2)
        self.assertEqual(self.adapter.calls, [])
        # live-armed with the focus lost: the still-moving frame injects
        # nothing either — the dispatch gate disarms instead
        self.ctl.set_practice(False)
        self.focus = False
        self.feed({'right': moved(point['right'], dx=.3)}, .25)
        self.assertFalse(self.ctl.armed)
        self.assertEqual(self.adapter.calls, [])

    def test_inventory_cursor_moves_only_while_the_hand_moves(self):
        self.engine.set_inventory_state(True)
        point = {'right': pose((0,))}
        self.feed(point, 0)
        self.feed(point, .15)
        self.feed(point, .2)  # settled and held still: nothing reaches the OS
        self.assertNotIn('move_relative', self.adapter.kinds())
        # a moving wrist injects one relative step per moving frame
        self.feed({'right': moved(point['right'], dx=.002)}, .25)
        self.feed({'right': moved(point['right'], dx=.004)}, .3)
        moves = [c for c in self.adapter.calls if c[0] == 'move_relative']
        self.assertEqual(moves, [('move_relative', (4, 0)),
                                 ('move_relative', (4, 0))])
        # the hand stops at the offset: the cursor stops with it
        self.feed({'right': moved(point['right'], dx=.004)}, .35)
        self.feed({'right': moved(point['right'], dx=.004)}, .4)
        moves = [c for c in self.adapter.calls if c[0] == 'move_relative']
        self.assertEqual(len(moves), 2)

    def test_inventory_cursor_speed_tuning_reaches_pointer_pixels(self):
        self.engine.set_inventory_state(True)
        point = {'right': pose((0,))}
        self.feed(point, 0)
        self.feed(point, .15)
        # default gain 80*5 = 400 px/palm: a .002-image (.01 palm) wrist
        # step injects a 4 px relative move
        hand = moved(point['right'], dx=.002)
        self.feed({'right': hand}, .2)
        self.assertIn(('move_relative', (4, 0)), self.adapter.calls)
        # tuning the engine gain scales the injected pixels accordingly
        self.dispatch.pose_engine.set_inventory_cursor_speed(240)
        self.feed({'right': moved(hand, dx=.002)}, .25)
        self.assertIn(('move_relative', (12, 0)), self.adapter.calls)


class MouseAdapterTests(unittest.TestCase):
    def test_timed_right_click_and_emergency_release(self):
        now = [0.]
        adapter = win32input.SendInputAdapter(clock=lambda: now[0])
        with mock.patch.object(win32input, '_send', return_value=1) as send:
            adapter.click_mouse('right')
            self.assertTrue(adapter.right_held)
            adapter.click_mouse('right')
            self.assertEqual(send.call_count, 1)
            adapter.cancel_pending()
            self.assertFalse(adapter.right_held)
            self.assertEqual(send.call_args.args[0].mi.dwFlags, win32input.MOUSEEVENTF_RIGHTUP)
            now[0] = 1.
            adapter.service_pending()
            self.assertEqual(send.call_count, 2)

    def test_hotbar_wheel_sign_and_relative_look(self):
        adapter = win32input.SendInputAdapter()
        with mock.patch.object(win32input, '_send', return_value=1) as send:
            adapter.scroll(-1)
            self.assertEqual(send.call_args.args[0].mi.mouseData, (-120) & 0xffffffff)
            adapter.move_relative(-10, 5)
            self.assertEqual(send.call_args.args[0].mi.dx, -10)
            self.assertEqual(send.call_args.args[0].mi.dy, 5)

    def test_timed_left_click_single_down_up_on_service(self):
        now = [0.]
        adapter = win32input.SendInputAdapter(clock=lambda: now[0])
        with mock.patch.object(win32input, '_send', return_value=1) as send:
            adapter.click_mouse('left')
            self.assertTrue(adapter.left_held)
            adapter.click_mouse('left')  # already down: no second press
            self.assertEqual(send.call_count, 1)
            now[0] = 1.
            adapter.service_pending()
            self.assertFalse(adapter.left_held)
        flags = [c.args[0].mi.dwFlags for c in send.call_args_list]
        self.assertEqual(flags, [win32input.MOUSEEVENTF_LEFTDOWN,
                                 win32input.MOUSEEVENTF_LEFTUP])

    def test_disarm_releases_pending_left_click(self):
        adapter = win32input.SendInputAdapter(clock=lambda: 0.)
        ctl = SafetyController(adapter, focus_check=lambda: True)
        ctl.set_practice(False)
        ctl.set_armed(True)
        with mock.patch.object(win32input, '_send', return_value=1) as send:
            adapter.click_mouse('left')
            self.assertTrue(adapter.left_held)
            ctl.set_armed(False)
            self.assertFalse(adapter.left_held)
        flags = [c.args[0].mi.dwFlags for c in send.call_args_list]
        self.assertEqual(flags, [win32input.MOUSEEVENTF_LEFTDOWN,
                                 win32input.MOUSEEVENTF_LEFTUP])

    def test_tracking_loss_releases_pending_cycle_click(self):
        adapter = win32input.SendInputAdapter(clock=lambda: 0.)
        ctl = SafetyController(adapter, focus_check=lambda: True)
        ctl.set_practice(False)
        ctl.set_armed(True)
        dispatch = PoseDispatcher(GestureEngine(), PalmContactRecognizer(), ctl)
        with mock.patch.object(win32input, '_send', return_value=1) as send:
            dispatch.dispatch({'right': thumb_hand(.9)}, .15, .15)
            dispatch.dispatch({'right': thumb_hand(.5)}, .2, .2)
            self.assertTrue(adapter.left_held)
            self.assertEqual(send.call_count, 1)  # one down, nothing else
            dispatch.tracking_lost(.25)
            self.assertFalse(adapter.left_held)
        flags = [c.args[0].mi.dwFlags for c in send.call_args_list]
        self.assertEqual(flags, [win32input.MOUSEEVENTF_LEFTDOWN,
                                 win32input.MOUSEEVENTF_LEFTUP])

    def test_promotion_cancels_pending_click_deadline(self):
        now = [0.]
        adapter = win32input.SendInputAdapter(clock=lambda: now[0])
        ctl = SafetyController(adapter, focus_check=lambda: True)
        ctl.set_practice(False)
        ctl.set_armed(True)
        dispatch = PoseDispatcher(GestureEngine(), PalmContactRecognizer(), ctl)
        with mock.patch.object(win32input, '_send', return_value=1) as send:
            dispatch.dispatch({'right': thumb_hand(.9)}, .15, .15)
            dispatch.dispatch({'right': thumb_hand(.5)}, .2, .2)   # click down
            self.assertTrue(adapter.left_held)
            dispatch.dispatch({'right': thumb_hand(.9)}, .35, .35)
            dispatch.dispatch({'right': thumb_hand(.5)}, .4, .4)   # promote
            # the hold takes ownership of the button: the click's 100ms
            # release deadline is cancelled, so servicing releases nothing
            now[0] = 1.
            adapter.service_pending()
            self.assertTrue(adapter.left_held)
            self.assertEqual(send.call_count, 1)  # still only the first DOWN
            dispatch.tracking_lost(1.05)
            self.assertFalse(adapter.left_held)
        flags = [c.args[0].mi.dwFlags for c in send.call_args_list]
        self.assertEqual(flags, [win32input.MOUSEEVENTF_LEFTDOWN,
                                 win32input.MOUSEEVENTF_LEFTUP])

    def test_right_click_focus_loss_releases_and_disarms(self):
        adapter = win32input.SendInputAdapter(clock=lambda: 0.)
        ctl = SafetyController(adapter, focus_check=lambda: False)
        ctl.set_practice(False)
        ctl.set_armed(True)
        with mock.patch.object(win32input, '_send', return_value=1):
            adapter.click_mouse('right')
            ctl.service_inputs()
        self.assertFalse(adapter.right_held)
        self.assertFalse(ctl.armed)


if __name__ == '__main__':
    unittest.main()
