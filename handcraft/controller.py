"""Safety controller: applies gesture events and palm-contact binding
results to an input adapter under arming / practice-mode / focus gating,
and guarantees that held inputs are released on any stop path.

The controller never decides *what* gesture happened (that is the pure
engine's and recognizer's job); it only decides *whether* an input may
reach the OS. Fail-closed throughout: releases are always honoured,
injections only while armed, live, and — checked immediately before every
single dispatch — with Minecraft Java focused. A real OS-level release is
only ever sent for a button/key this controller actually pressed, or one
the adapter itself still tracks (e.g. a tap key stuck after a failed
cleanup).

:class:`ArmSequencer` implements the live-mode arming ritual: pressing Arm
starts a countdown (time to alt-tab into Minecraft); when it elapses the
controller is armed, but output stays gated on real Minecraft focus at
dispatch time. Any focus loss after focus was achieved disarms and demands
an explicit re-arm.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Set

from .adapters import InputAdapter, VK_E, vk_for_key
from .bindings import RecognitionResult
from .gestures import GestureEvent, GestureKind

ARM_COUNTDOWN_S = 3.0


class ArmSequencer:
    """Pure arming state machine (no I/O, no clock reads).

    Phases: ``idle`` -> ``counting`` -> ``armed``. The app feeds
    :meth:`poll` a monotonic clock; when the countdown elapses it returns
    True exactly once and the phase becomes ``armed``.
    """

    def __init__(self, countdown_s: float = ARM_COUNTDOWN_S):
        self.countdown_s = float(countdown_s)
        self.phase = "idle"
        self._deadline = 0.0

    @property
    def counting(self) -> bool:
        return self.phase == "counting"

    @property
    def armed(self) -> bool:
        return self.phase == "armed"

    def request(self, now: float) -> None:
        self.phase = "counting"
        self._deadline = now + self.countdown_s

    def cancel(self) -> None:
        self.phase = "idle"

    def poll(self, now: float) -> bool:
        """True exactly once, when a running countdown completes."""
        if self.phase == "counting" and now >= self._deadline:
            self.phase = "armed"
            return True
        return False

    def remaining(self, now: float) -> float:
        if self.phase != "counting":
            return 0.0
        return max(0.0, self._deadline - now)


class SafetyController:
    def __init__(self, adapter: InputAdapter,
                 on_action: Optional[Callable[[str], None]] = None,
                 focus_check: Optional[Callable[[], bool]] = None):
        self._adapter = adapter
        self._on_action = on_action or (lambda msg: None)
        self._focus_check = focus_check
        self.armed = False
        self.practice = True
        self._held = False           # logical mining-hold state (status display)
        self._os_held = False        # we really pressed the OS mouse button
        self._keys_held: Set[int] = set()      # OS keys we really pressed
        self._practice_keys: Set[str] = set()  # logical holds, for practice log

    # ------------------------------------------------------------ properties

    @property
    def mouse_held(self) -> bool:
        return self._held

    @property
    def held_keys(self) -> Set[int]:
        return set(self._keys_held)

    # ---------------------------------------------------------------- control

    def set_armed(self, armed: bool) -> None:
        if armed:
            self.armed = True
        else:
            # Disarm first: even if the cleanup below raises, the
            # controller must never stay armed.
            self.armed = False
            self.release_all("disarm")
        self._log("ARMED" if armed else "DISARMED")

    def set_practice(self, practice: bool) -> None:
        if practice:
            self.release_all("practice-on")
        self.practice = practice
        self._practice_keys.clear()
        self._log("PRACTICE MODE" if practice else "LIVE MODE")

    def release_all(self, reason: str = "release") -> None:
        """Total input release. Idempotent; attempts every release even
        when some fail, keeps failed holds tracked so a later call retries
        them, and asks the adapter to release anything it tracks on its own
        (e.g. a key left down by a tap whose cleanup failed)."""
        if self._os_held:
            try:
                self._adapter.left_mouse_up()
            except Exception:
                self._log(f"left mouse release FAILED ({reason}) - still tracked")
            else:
                self._os_held = False
                self._log(f"left mouse RELEASED ({reason})")
        self._held = False
        for vk in sorted(self._keys_held):
            try:
                self._adapter.key_up(vk)
            except Exception:
                self._log(f"key 0x{vk:02X} release FAILED ({reason}) - still tracked")
                continue
            self._keys_held.discard(vk)
            self._log(f"key 0x{vk:02X} RELEASED ({reason})")
        # Adapter-only pending state (e.g. a tap key stuck after a failed
        # cleanup) is known to the adapter, not to us: let it retry.
        try:
            adapter_busy = bool(getattr(self._adapter, "held_keys", None)) or \
                bool(getattr(self._adapter, "left_held", False))
            adapter_busy = adapter_busy or bool(getattr(self._adapter, "right_held", False))
        except Exception:
            adapter_busy = False
        if adapter_busy:
            try:
                self._adapter.release_all()
            except Exception:
                self._log(f"adapter release_all FAILED ({reason})")
        self._practice_keys.clear()

    # ---------------------------------------------------------------- events

    def service_inputs(self) -> None:
        """Maintain timed input pulses even when no tracking frame arrives."""
        if (getattr(self._adapter, "held_keys", None)
                or getattr(self._adapter, "left_held", False)
                or getattr(self._adapter, "right_held", False)):
            if not self.armed or self.practice:
                self.release_all("inactive pulse")
                return
            if not self._focus_gate():
                return
        try:
            self._adapter.service_pending()
        except Exception:
            self._fail_closed("timed pulse release failed")
            raise

    def handle(self, events: Iterable[GestureEvent]) -> None:
        for event in events:
            self._dispatch(event)

    def apply_pointer(self, result) -> None:
        """Pose-driven look, inventory cursor, use clicks and hotbar wheel."""
        actions = []
        if result.cursor is not None:
            actions.append(("move_cursor", result.cursor))
        elif result.look_dx or result.look_dy:
            actions.append(("move_relative", (round(result.look_dx), round(result.look_dy))))
        if result.click:
            actions.append(("click_mouse", (result.click,)))
        if result.scroll:
            actions.append(("scroll", (result.scroll,)))
        if not self.armed:
            return
        for name, args in actions:
            if self.practice:
                if name != "move_relative" and name != "move_cursor":
                    self._log(f"[practice] {name}: {args}")
                continue
            if not self._focus_gate():
                return
            try:
                getattr(self._adapter, name)(*args)
            except Exception:
                self._fail_closed(f"{name} failed")
                raise
            if name in {"click_mouse", "scroll"}:
                self._log(f"{name}: {args}")

    def apply_bindings(self, result: RecognitionResult,
                       movement_enabled: bool = True) -> None:
        """Apply one palm-contact recognition frame.

        Holds are diffed against the keys currently down: missing keys are
        released (always honoured, fail closed), new keys pressed only
        through the armed/live/focused gate. Pulses fire once per contact.
        ``movement_enabled=False`` (inventory believed open) suppresses
        held movement keys and pulses entirely and releases any held
        movement keys that were down.
        """
        target: Set[int] = set()
        if movement_enabled:
            for key in sorted(result.held_keys):
                vk = vk_for_key(key)
                if vk is not None:
                    target.add(vk)
        pulses = tuple(result.pulse_keys) if movement_enabled else ()
        if not movement_enabled or not result.tracked_hands:
            cancel = getattr(self._adapter, "cancel_pending", None)
            if cancel is not None:
                try:
                    cancel()
                except Exception:
                    self._fail_closed("timed pulse cancellation failed")
                    raise

        # Releases first — always honoured, even disarmed or in practice.
        # A failed release disarms and cleans up before re-raising; the
        # key stays tracked so a later release_all retries it.
        for vk in sorted(self._keys_held - target):
            try:
                self._adapter.key_up(vk)
            except Exception:
                self._fail_closed(f"key 0x{vk:02X} release failed")
                raise
            self._keys_held.discard(vk)
            self._log(f"key 0x{vk:02X} RELEASED (contact ended)")

        held_log = sorted(result.held_keys) if movement_enabled else []
        if self.practice:
            for key in held_log:
                if key not in self._practice_keys:
                    self._log(f"[practice] hold {key}")
            self._practice_keys = set(held_log)
            for key in pulses:
                self._log(f"[practice] pulse {key}")
            return

        if not self.armed:
            return

        # Frame-level gate: every live frame that holds or pulses anything
        # is focus-checked, even when the held keys are unchanged. Focus
        # loss releases everything and disarms.
        if target or pulses:
            if not self._focus_gate():
                return

        for vk in sorted(target - self._keys_held):
            if not self._focus_gate():
                return
            try:
                self._adapter.key_down(vk)
            except Exception:
                self._fail_closed("key_down failed")
                raise
            self._keys_held.add(vk)
            self._log(f"key 0x{vk:02X} DOWN (binding hold)")

        for key in pulses:
            vk = vk_for_key(key)
            if vk is None:
                self._log(f"unknown key {key!r} - skipped")
                continue
            if vk in self._keys_held:
                self._log(f"{key} pulse skipped - key already held")
                continue
            if not self._focus_gate():
                return
            try:
                self._adapter.tap_key(vk)
            except Exception:
                self._fail_closed("tap_key failed")
                raise
            self._log(f"{key} tapped (binding pulse)")

    # ------------------------------------------------------------- dispatch

    def _dispatch(self, event: GestureEvent) -> None:
        kind = event.kind

        # Releases are always honoured, even while disarmed — fail closed.
        # STOP_ATTACK releases the mouse only: held movement contacts
        # (binding keys) keep walking.
        if kind is GestureKind.STOP_ATTACK:
            if self._held or self._os_held:
                self._release_mouse("lease/stop")
            return

        if not self.armed:
            return

        if kind is GestureKind.PUNCH:
            # Held lease: repeated PUNCHs keep the button held — never a
            # short click that would reset mining.
            if self.practice:
                if not self._held:
                    self._log("[practice] punch -> hold left mouse")
                self._held = True
            else:
                if not self._os_held:
                    if not self._focus_gate():
                        return
                    try:
                        self._adapter.left_mouse_down()
                    except Exception:
                        self._fail_closed("left_mouse_down failed")
                        raise
                    self._os_held = True
                    self._log("left mouse DOWN (mining hold)")
                self._held = True

        elif kind is GestureKind.OPEN_INVENTORY:
            if self.practice:
                self._log("[practice] open inventory -> tap E")
            else:
                if not self._focus_gate():
                    return
                try:
                    self._adapter.tap_key(VK_E)
                except Exception:
                    self._fail_closed("tap_key failed")
                    raise
                self._log("E tapped (open inventory)")

        elif kind is GestureKind.CLOSE_INVENTORY:
            if self.practice:
                self._log("[practice] close inventory -> tap E")
            else:
                if not self._focus_gate():
                    return
                try:
                    self._adapter.tap_key(VK_E)
                except Exception:
                    self._fail_closed("tap_key failed")
                    raise
                self._log("E tapped (close inventory)")

    # ---------------------------------------------------------------- misc

    def _release_mouse(self, reason: str) -> None:
        """Release only the mining-hold mouse button. A failed release
        keeps ``_os_held`` set so :meth:`release_all` retries it, disarms
        and cleans up via :meth:`_fail_closed`, then re-raises."""
        self._held = False
        if not self._os_held:
            return
        try:
            self._adapter.left_mouse_up()
        except Exception:
            self._fail_closed(f"left mouse release failed ({reason})")
            raise
        self._os_held = False
        self._log(f"left mouse RELEASED ({reason})")

    def _fail_closed(self, reason: str) -> None:
        """An input injection failed: disarm *before* cleanup, attempt to
        release everything, keep failed bookkeeping. The caller re-raises
        the original error."""
        self.armed = False
        try:
            self.release_all(reason)
        except Exception:
            pass
        self._log(f"INPUT FAILURE ({reason}) - released and DISARMED")

    def _focus_gate(self) -> bool:
        """Checked immediately before every live input dispatch.

        No cache: the callable is invoked now. Any failure releases
        everything and disarms so an explicit re-arm is required.
        """
        if self._focus_check is None:
            return True
        try:
            focused = bool(self._focus_check())
        except Exception:
            focused = False
        if focused:
            return True
        # Disarm *before* cleanup: even if a release raises, the
        # controller must never stay armed after focus is lost.
        self.armed = False
        try:
            self.release_all("focus lost at dispatch")
        except Exception:
            pass
        self._log("FOCUS LOST at dispatch - released and DISARMED")
        return False

    def _log(self, msg: str) -> None:
        try:
            self._on_action(msg)
        except Exception:
            pass  # logging must never break input safety
