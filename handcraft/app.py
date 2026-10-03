"""Tkinter GUI for Minecraft Hand Detector.

All Tk work happens on the main thread. The camera worker communicates
through bounded queues; a ``root.after`` tick drains them (newest result
only), feeds the pure gesture engine (right hand: mining + inventory) and
the palm-contact recognizer (left hand: W hold / Space pulse bindings),
and applies results through the safety controller.

Safety wiring:
* Practice mode is ON by default — no real input is ever injected until
  the user switches to live mode AND completes the arming countdown.
* Switching practice mode either way disarms and cancels any countdown
  FIRST: a practice-armed state can never become live-armed directly.
* Live Arm disarms first, refuses while the inventory belief is UNKNOWN
  (a manual Mark open/closed sync is required), preserves that known
  belief across the reset, then starts a 3-second countdown so the user
  can switch to Minecraft; output stays gated on an immediate
  Minecraft-Java focus check before every single input dispatch. While
  live-disarmed, unfocused, or inventory-unknown, gesture results never
  advance the right-hand game state nor fire bindings — the recognizer
  still updates for the UI and held inputs are released. Once focus has
  been achieved, any subsequent focus loss releases inputs, disarms,
  marks the inventory belief UNKNOWN (manual sync required) and demands
  an explicit re-arm.
* F8 is a global emergency stop polled from the Tk loop — it cancels any
  in-progress model download and stops the worker even when input release
  fails, releases inputs, disarms and stops tracking.
* Result packets carry their pre-inference capture timestamp; packets
  older than 300 ms of total capture age release all holds via
  tracking-loss, and >300 ms without a fresh capture also feeds a
  tracking-loss update so held inputs release.
* All queues are cleared on every start/stop, worker status messages are
  stamped with the capture generation, and results are only accepted
  while a worker of the current generation is active, so stale packets or
  errors can never affect a new capture session.
* The gesture configuration panel opens as a Toplevel attached to this
  root (never a second Tk), shows both live hands with numbered joints,
  and saving it reloads the recognizer bindings.
"""

from __future__ import annotations

import os
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, List, Optional

from . import cameras
from .adapters import NullAdapter, vk_for_key
from .bindings import PalmContactRecognizer, RecognitionResult
from .controller import ArmSequencer, SafetyController
from .focus import emergency_stop_pressed, is_windows, minecraft_java_focused
from .gestures import GestureEngine
from .geometry import Point
from .tracker import CameraWorker, DownloadCancelled, ensure_model

STALE_FRAME_S = 0.30
TICK_MS = 8
FOCUS_CHECK_MS = 250
F8_CHECK_MS = 80

BG = "#1e1f24"
PANEL = "#2a2c33"
FG = "#e6e6e6"
ACCENT = "#4f8cff"

STATUS_COLORS = {
    "idle": "#8a8f98",
    "practice": "#e6b93d",
    "armed": "#3fca6b",
    "alert": "#e05252",
}

LEGEND_TEXT = (
    "Status legend:  "
    "gray = idle / stopped   yellow = practice or disarmed   "
    "green = ARMED (live inputs)   red = emergency / focus lost / error"
)

# Status-packet kinds from the preparation thread.
_STATUS_WORKER_READY = "worker_ready"
_STATUS_PREP_FAILED = "prep_failed"
_STATUS_PREP_CANCELLED = "prep_cancelled"
_STATUS_WORKER_MSG = "worker_msg"


class ResultDispatcher:
    """Tk-free result pipeline, unit-testable without a GUI.

    Freshness: a packet whose pre-inference capture timestamp is older
    than ``max_age_s`` is discarded before any engine/recognizer/controller
    call. Fresh packets run the right hand through the gesture engine
    (mining + inventory only) and both hands through the palm-contact
    recognizer (bindings); movement holds are suppressed while the
    inventory is believed open.

    Gating: while ``gate`` is supplied and returns False (live mode but
    disarmed, Minecraft unfocused, or inventory belief UNKNOWN), a fresh
    packet must NOT advance the right-hand game state nor send binding
    actions — otherwise disarmed gestures could set the inventory belief
    without E ever being pressed, or punch while the belief is unknown.
    The recognizer still updates (``last_result`` feeds the config panel
    UI), and anything currently held is released fail-closed. The gate is
    never consulted for releases.
    """

    def __init__(self, engine: GestureEngine,
                 recognizer: PalmContactRecognizer,
                 controller: SafetyController,
                 max_age_s: float = STALE_FRAME_S,
                 gate: Optional[Callable[[], bool]] = None):
        self.engine = engine
        self.recognizer = recognizer
        self.controller = controller
        self.max_age_s = max_age_s
        self._gate = gate
        self.last_result: Optional[RecognitionResult] = None

    def _gated(self) -> bool:
        return self._gate is not None and not self._gate()

    def dispatch(self, hands: Optional[Dict[str, List[Point]]],
                 captured_at: float, now: float) -> bool:
        """Process one packet. Returns False (no action taken) if stale.

        A state change is only *captured* when live focus held through the
        whole dispatch: if the frame started live-armed and the controller
        ended up disarmed (focus or input failure at dispatch time), the
        engine may have advanced on state whose input never left the
        machine — the inventory belief is invalidated to UNKNOWN and
        transient engine tracking is reset, so nothing unsent is treated
        as known."""
        if now - captured_at > self.max_age_s:
            return False
        hands = hands or {}
        if self._gated():
            # Live-disarmed / unfocused / inventory-unknown: do not advance
            # game state; release anything held; UI still sees the frame.
            self.controller.handle(self.engine.update(None, captured_at))
            result = self.recognizer.update_hands(hands)
            self.last_result = result
            self.controller.apply_bindings(result, movement_enabled=False)
            return True
        live_armed = self.controller.armed and not self.controller.practice
        try:
            right = hands.get("right")
            self.controller.handle(self.engine.update(right, captured_at))
            result = self.recognizer.update_hands(hands)
            self.last_result = result
            self.controller.apply_bindings(
                result,
                movement_enabled=self.engine.inventory_believed is not True)
        finally:
            if live_armed and not self.controller.armed:
                # Focus/input failure mid-dispatch: the E tap (or a hold)
                # was blocked or blew up AFTER the engine advanced.
                try:
                    self.controller.handle(self.engine.reset())
                except Exception:
                    pass  # already disarmed; any original error propagates
                self.engine.set_inventory_state(None)
        return True

    def tracking_lost(self, now: float) -> None:
        """Tracking loss: engine sees no hand, recognizer sees no hands —
        both release whatever they were holding."""
        self.controller.handle(self.engine.update(None, now))
        result = self.recognizer.update_hands({})
        self.last_result = result
        self.controller.apply_bindings(result, movement_enabled=False)

    def reset(self, preserve_inventory: bool = False) -> None:
        """Mode/arm transitions: recognizers and inventory belief reset so
        practice events never carry into live. ``preserve_inventory=True``
        keeps a manually synced known belief across the reset (live Arm
        requires and preserves it)."""
        self.controller.handle(self.engine.reset())
        if not preserve_inventory:
            self.engine.set_inventory_state(None)
        self.recognizer.reset()
        self.last_result = None


class HandCraftApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Minecraft Hand Detector")
        root.configure(bg=BG)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = os.path.join(base_dir, "models")

        self.engine = GestureEngine()
        self.recognizer = PalmContactRecognizer()
        self.adapter = self._make_adapter()
        self.controller = SafetyController(
            self.adapter,
            on_action=self._log_action,
            focus_check=minecraft_java_focused)
        from .posedispatch import PoseDispatcher
        self.dispatcher = PoseDispatcher(
            self.engine, self.recognizer, self.controller,
            gate=self._gestures_live_enabled)
        self.sequencer = ArmSequencer()

        self.frame_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=2)
        self.result_queue: "queue.Queue" = queue.Queue(maxsize=4)
        self.status_queue: "queue.Queue" = queue.Queue()

        self.worker: Optional[CameraWorker] = None
        self._generation = 0              # bumped on every start/stop
        self._prep_stop: Optional[threading.Event] = None
        self._preparing = False
        self._last_capture_at: Optional[float] = None  # capture clock, not processing time
        self._f8_was_down = False
        self._focus_ok = False
        self._focus_achieved = False      # focus seen since arming
        self._photo = None                # keep PhotoImage alive
        self._latest_hands: Dict[str, List[Point]] = {}

        # Gesture configuration panel (Toplevel attached to this root).
        from . import handmap
        self._handmap = handmap
        self._config_path = handmap.default_config_path()
        self._config_panel = None
        self._guide = None
        self._config_mtime: Optional[float] = self._config_file_mtime()

        self._reload_recognizer()         # pick up saved bindings
        self._camera_config_path = cameras.default_camera_config_path()
        self._camera_devices: List[cameras.CameraDevice] = []
        self._build_ui()
        self._tick()
        self._focus_tick()
        self._f8_tick()

    # ------------------------------------------------------------ UI build

    def _build_ui(self) -> None:
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=10, pady=10)

        # Left: camera preview
        left = tk.Frame(main, bg=BG)
        left.pack(side="left", fill="both", expand=True)
        self.preview = tk.Label(left, bg="#101116", width=60, height=22,
                                text="camera preview", fg="#666a73")
        self.preview.pack(fill="both", expand=True)

        # Right: controls
        right = tk.Frame(main, bg=PANEL, padx=12, pady=12)
        right.pack(side="right", fill="y", padx=(10, 0))

        tk.Label(right, text="Minecraft Hand Detector", bg=PANEL, fg=FG,
                 font=("Segoe UI", 14, "bold")).pack(anchor="w")

        self.status_dot = tk.Canvas(right, width=16, height=16, bg=PANEL,
                                    highlightthickness=0)
        self.status_dot.pack(anchor="w", pady=(10, 0))
        self._dot = self.status_dot.create_oval(2, 2, 14, 14,
                                                fill=STATUS_COLORS["idle"],
                                                outline="")
        self.status_label = tk.Label(right, text="", bg=PANEL, fg=FG,
                                     wraplength=220, justify="left")
        self.status_label.pack(anchor="w")

        ttk.Separator(right).pack(fill="x", pady=8)

        tk.Label(right, text="Camera (pick your phone, e.g. Xiaomi):",
                 bg=PANEL, fg=FG).pack(anchor="w")
        cam_row = tk.Frame(right, bg=PANEL)
        cam_row.pack(fill="x")
        self.camera_var = tk.StringVar(value="")
        self.camera_box = ttk.Combobox(
            cam_row, textvariable=self.camera_var, width=30, state="readonly",
            values=())
        self.camera_box.pack(side="left", fill="x", expand=True)
        self.camera_box.bind("<<ComboboxSelected>>", self._on_camera_selected)
        tk.Button(cam_row, text="Refresh", command=self._refresh_cameras,
                  bg="#3a3d46", fg=FG, relief="flat").pack(side="left",
                                                          padx=(6, 0))

        self.start_btn = tk.Button(right, text="Start tracking",
                                   command=self.start_tracking,
                                   bg=ACCENT, fg="white", relief="flat")
        self.start_btn.pack(fill="x", pady=(8, 2))
        self.stop_btn = tk.Button(right, text="Stop tracking",
                                  command=self.stop_tracking,
                                  bg=PANEL, fg=FG, relief="flat",
                                  state="disabled")
        self.stop_btn.pack(fill="x", pady=2)

        ttk.Separator(right).pack(fill="x", pady=8)

        tk.Button(right, text="Gesture guide (keep beside game)",
                  command=self._open_guide,
                  bg="#3a3d46", fg=FG, relief="flat").pack(fill="x", pady=2)
        tk.Button(right, text="Calibrate neutral hand positions",
                  command=self._calibrate_poses,
                  bg="#3a3d46", fg=FG, relief="flat").pack(fill="x", pady=2)

        ttk.Separator(right).pack(fill="x", pady=8)

        self.practice_var = tk.BooleanVar(value=True)
        tk.Checkbutton(right, text="Practice mode (no real inputs)",
                       variable=self.practice_var, command=self._on_practice,
                       bg=PANEL, fg=FG, selectcolor=PANEL,
                       activebackground=PANEL).pack(anchor="w")

        self.arm_btn = tk.Button(right, text="Arm", command=self._on_arm,
                                 bg="#3a3d46", fg=FG, relief="flat")
        self.arm_btn.pack(fill="x", pady=(8, 2))
        self.disarm_btn = tk.Button(right, text="Disarm",
                                    command=self._on_disarm,
                                    bg=PANEL, fg=FG, relief="flat")
        self.disarm_btn.pack(fill="x", pady=2)

        ttk.Separator(right).pack(fill="x", pady=8)

        tk.Label(right, text="Sensitivity", bg=PANEL, fg=FG).pack(anchor="w")
        self.sens_var = tk.DoubleVar(value=1.0)
        tk.Scale(right, from_=0.5, to=2.0, resolution=0.1,
                 orient="horizontal", variable=self.sens_var,
                 command=self._on_sensitivity, bg=PANEL, fg=FG,
                 troughcolor="#3a3d46", highlightthickness=0,
                 length=200).pack(fill="x")

        ttk.Separator(right).pack(fill="x", pady=8)

        tk.Label(right,
                 text="Inventory cursor speed (px/sec) - lower = slower",
                 bg=PANEL, fg=FG).pack(anchor="w")
        self.inventory_cursor_speed_var = tk.IntVar(value=80)
        tk.Scale(right, from_=20, to=240, resolution=10,
                 orient="horizontal",
                 variable=self.inventory_cursor_speed_var,
                 command=self._on_inventory_cursor_speed, bg=PANEL, fg=FG,
                 troughcolor="#3a3d46", highlightthickness=0,
                 length=200).pack(fill="x")

        ttk.Separator(right).pack(fill="x", pady=8)

        self.inv_label = tk.Label(right, text="Inventory belief: closed",
                                  bg=PANEL, fg=FG)
        self.inv_label.pack(anchor="w")
        inv_row = tk.Frame(right, bg=PANEL)
        inv_row.pack(fill="x", pady=4)
        tk.Button(inv_row, text="Mark open",
                  command=lambda: self._sync_inventory(True),
                  bg="#3a3d46", fg=FG, relief="flat").pack(side="left",
                                                           expand=True,
                                                           fill="x")
        tk.Button(inv_row, text="Mark closed",
                  command=lambda: self._sync_inventory(False),
                  bg="#3a3d46", fg=FG, relief="flat").pack(side="left",
                                                           expand=True,
                                                           fill="x",
                                                           padx=(4, 0))

        tk.Label(right, text="F8 = emergency stop (works anywhere)",
                 bg=PANEL, fg="#e05252").pack(anchor="w", pady=(10, 0))

        # Bottom: legend + log
        bottom = tk.Frame(self.root, bg=BG)
        bottom.pack(fill="both", padx=10, pady=(0, 10))
        tk.Label(bottom, text=LEGEND_TEXT, bg=BG, fg="#9aa0aa",
                 anchor="w").pack(fill="x")
        self.log = tk.Text(bottom, height=6, bg="#101116", fg="#b9bec7",
                           state="disabled", relief="flat")
        self.log.pack(fill="both", expand=True, pady=(6, 0))

        self._set_status("idle", "Idle - press Start to begin tracking")
        self._refresh_cameras(initial=True)

    # ------------------------------------------------------------ cameras

    def _refresh_cameras(self, initial: bool = False) -> None:
        """Re-enumerate named cameras (never opens a device) and refill the
        dropdown. Selection priority: saved (initial launch) or current
        (Refresh) stable identity -> Xiaomi-family name -> none. There is
        NO positional fallback: a missing phone never silently selects the
        integrated laptop webcam."""
        if self.worker is not None or self._preparing:
            self.stop_tracking()  # releases inputs and disarms first
        # Capture the current selection's stable identity BEFORE enumeration
        # replaces the device list: the old dropdown position is meaningless
        # against the new list and must never be mapped onto a new device.
        prior = None if initial else self._selected_camera_device()
        try:
            devices = cameras.enumerate_camera_devices()
            error = None
        except Exception as exc:
            devices, error = [], exc
        self._camera_devices = devices
        self.camera_box.configure(values=[d.display_name for d in devices])
        if error is not None:
            self.camera_var.set("")
            self._set_status("alert", f"Camera enumeration failed: {error}")
            return
        if not devices:
            self.camera_var.set("")
            self._set_status(
                "alert",
                "No cameras found. Connect the phone camera (Xiaomi "
                "connected-camera / Windows phone camera) and press Refresh.")
            return

        saved = None
        if initial:
            saved = cameras.load_saved_camera(self._camera_config_path)
        elif prior is not None:
            saved = {"identity": prior.identity, "name": prior.name}
        chosen = None
        if saved:
            chosen = cameras.resolve_device(
                devices, identity=saved.get("identity"),
                name=saved.get("name"))
        if chosen is None:
            chosen = cameras.prefer_device(devices)
        if chosen is None:
            self.camera_var.set("")
            if saved:
                self._set_status(
                    "alert",
                    f"Saved camera {saved.get('name', '?')!r} not found. "
                    "Reconnect it and press Refresh, or pick another camera "
                    "from the list.")
            else:
                self._set_status(
                    "practice",
                    "Select your phone camera from the Camera list "
                    "(full device names), then press Start.")
            return
        self.camera_box.current(devices.index(chosen))
        if not saved or saved.get("identity") != chosen.identity:
            try:
                cameras.save_camera(chosen, self._camera_config_path)
            except Exception as exc:
                self._log_action(f"camera config save FAILED: {exc}")
            self._log_action(f"camera selected: {chosen.display_name}")

    def _selected_camera_device(self) -> Optional[cameras.CameraDevice]:
        devices = getattr(self, "_camera_devices", None) or []
        box = getattr(self, "camera_box", None)
        try:
            idx = box.current() if box is not None else -1
        except Exception:
            idx = -1
        if 0 <= idx < len(devices):
            return devices[idx]
        return None

    def _on_camera_selected(self, _event=None) -> None:
        """Manual named selection: stop tracking first (releases inputs and
        disarms), then persist the stable identity."""
        if self.worker is not None or self._preparing:
            self.stop_tracking()
        device = self._selected_camera_device()
        if device is None:
            return
        try:
            cameras.save_camera(device, self._camera_config_path)
        except Exception as exc:
            self._log_action(f"camera config save FAILED: {exc}")
        self._log_action(f"camera selected: {device.display_name}")

    def _current_camera_request(self):
        """The CameraDevice for the dropdown selection (re-resolved against
        a fresh enumeration in the preparation thread), or a plain int index
        for legacy bare-shell callers. Returns None — with an actionable
        status — when there is no valid named selection; never a silent
        index-0/laptop fallback."""
        devices = getattr(self, "_camera_devices", None)
        if devices is None:  # bare test shell: explicit numeric index
            try:
                return int(self.camera_var.get())
            except Exception:
                return None  # no silent index-0 fallback on invalid input
        device = self._selected_camera_device()
        if device is not None:
            return device
        if not devices:
            self._set_status(
                "alert",
                "No camera found. Connect the phone camera and press "
                "Refresh, then Start.")
        else:
            self._set_status(
                "alert",
                "No camera selected. Pick your phone camera (e.g. Xiaomi) "
                "from the Camera list, then press Start.")
        return None

    # ------------------------------------------------------------- helpers

    def _make_adapter(self):
        if is_windows():
            try:
                from .win32input import SendInputAdapter
                return SendInputAdapter()
            except Exception as exc:
                self._defer_log = f"SendInput unavailable ({exc}); live mode disabled"
        return NullAdapter()

    def _log_action(self, msg: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", time.strftime("%H:%M:%S ") + msg + "\n")
        lines = int(self.log.index("end-1c").split(".")[0])
        if lines > 200:
            self.log.delete("1.0", f"{lines - 200}.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _set_status(self, level: str, text: str) -> None:
        self.status_dot.itemconfigure(self._dot, fill=STATUS_COLORS[level])
        self.status_label.configure(text=text)

    # ----------------------------------------------------- configuration UI

    def _config_file_mtime(self) -> Optional[float]:
        try:
            return os.path.getmtime(self._config_path)
        except OSError:
            return None

    def _reload_recognizer(self) -> None:
        """(Re)build the palm-contact recognizer from the saved bindings.

        Called from ``__init__`` BEFORE the log widget exists, so a
        failure warning goes to ``self._defer_log`` (consumed once the UI
        is up) instead of ``self._log_action``. The load, the full-schema
        ``PalmContactRecognizer`` construction, and the ``vk_for_key``
        supported-key check all run inside one guard: on ANY error a
        DEFAULT recognizer is built instead, the corrupt config file is
        left untouched, and an invalid config can never arm a binding or
        inject input."""
        try:
            config = self._handmap.load_binding_config(self._config_path)
            bindings = config["bindings"]
            recognizer = PalmContactRecognizer(bindings=bindings)
            for binding in recognizer.bindings:
                if vk_for_key(binding.key) is None:
                    raise ValueError(
                        f"binding {binding.gesture_id!r} key "
                        f"{binding.key!r} has no virtual-key mapping")
        except Exception as exc:
            warning = f"binding config invalid ({exc}); using defaults"
            if hasattr(self, "log"):
                self._log_action(warning)
            else:
                self._defer_log = warning
            recognizer = PalmContactRecognizer()  # safe defaults
        self.recognizer = recognizer
        self.dispatcher.recognizer = self.recognizer
        self._config_mtime = self._config_file_mtime()

    def _poll_config_save(self) -> None:
        """Saving the panel rewrites the JSON; reload the recognizer then."""
        mtime = self._config_file_mtime()
        if mtime is not None and mtime != self._config_mtime:
            self._reload_recognizer()
            self._log_action("gesture bindings reloaded from saved config")

    def _open_config_panel(self) -> None:
        """Open the hand-map configuration panel as a Toplevel attached to
        this root — never a second Tk instance, never monkeypatched."""
        panel = self._config_panel
        if panel is not None:
            try:
                if panel.winfo_exists():
                    panel.lift()
                    panel.focus_force()
                    return
            except Exception:
                pass
            self._config_panel = None

        panel = self._handmap.GestureConfigPanel(
            parent=self.root,
            config_path=self._config_path,
            on_bindings_changed=self._on_bindings_changed)
        self._config_panel = panel
        panel.protocol("WM_DELETE_WINDOW", self._close_config_panel)

    def _on_bindings_changed(self, bindings) -> None:
        """The panel saved new bindings: release anything held, reset
        contact state, and reload the recognizer from the saved config."""
        try:
            self.controller.release_all("bindings changed")
        except Exception as exc:
            self._log_action(f"release after binding save FAILED: {exc}")
        self.dispatcher.last_result = None
        self._reload_recognizer()
        self._config_mtime = self._config_file_mtime()
        self._log_action("gesture bindings reloaded from saved config")

    def _close_config_panel(self) -> None:
        panel = self._config_panel
        self._config_panel = None
        if panel is not None:
            try:
                panel.destroy()
            except Exception:
                pass
        self._poll_config_save()

    def _update_config_panel(self) -> None:
        """Live-update the panel's hand map and contact states, and pick up
        saved binding changes. Passes the dispatcher's last result (never
        the recognizer itself, which would consume pulse state twice)."""
        panel = self._config_panel
        if panel is None:
            return
        try:
            if not panel.winfo_exists():
                self._config_panel = None
                return
            panel.update_hands(self._latest_hands, self.dispatcher.last_result)
        except tk.TclError:
            self._config_panel = None
        self._poll_config_save()

    # ------------------------------------------------------------- controls

    def start_tracking(self) -> None:
        if self.worker is not None or self._preparing:
            return
        camera = self._current_camera_request()
        if camera is None:
            return  # actionable status already shown; no fallback
        self._generation += 1
        generation = self._generation
        self._clear_queues()  # no stale packets/errors in a new session
        self._prep_stop = threading.Event()
        self._preparing = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")  # Stop/F8 cancel preparation
        self._set_status("practice", "Preparing model (downloads once)...")
        threading.Thread(target=self._prepare_and_start,
                         args=(generation, camera, self._prep_stop),
                         daemon=True).start()

    def _clear_queues(self) -> None:
        """Drop every queued frame/result/status packet (start/stop)."""
        for q in (self.frame_queue, self.result_queue, self.status_queue):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def _prepare_and_start(self, generation: int, camera,
                           stop_event: threading.Event) -> None:
        """Preparation thread: cancellable model download, then hand the
        worker to the main thread via a status packet. The worker object
        is assigned and started ONLY on the main thread, and only if the
        generation still matches (a Stop bumps it), so a cancelled download
        can never restart the camera.

        ``camera`` is a cameras.CameraDevice from the UI: the selected
        device is re-resolved against a FRESH enumeration here (by stable
        identity, never by remembered index) so numeric drift between
        selection and Start cannot open the wrong camera. A plain int is
        accepted for legacy/test callers (default backend). A selection
        that no longer exists fails with an actionable error — there is no
        fallback to any other camera."""
        try:
            model_path = ensure_model(
                self.models_dir,
                status=lambda m: self.status_queue.put(m),
                stop_event=stop_event)
        except DownloadCancelled:
            self.status_queue.put((_STATUS_PREP_CANCELLED, generation))
            return
        except Exception as exc:
            self.status_queue.put(
                (_STATUS_PREP_FAILED, generation,
                 f"ERROR: model download failed: {exc}"))
            return
        camera_backend = None
        camera_name = None
        if isinstance(camera, cameras.CameraDevice):
            if stop_event.is_set():
                self.status_queue.put((_STATUS_PREP_CANCELLED, generation))
                return
            try:
                devices = cameras.enumerate_camera_devices()
            except Exception as exc:
                self.status_queue.put(
                    (_STATUS_PREP_FAILED, generation,
                     f"ERROR: camera enumeration failed: {exc}"))
                return
            resolved = cameras.resolve_device(
                devices, identity=camera.identity, name=camera.name)
            if resolved is None:
                self.status_queue.put(
                    (_STATUS_PREP_FAILED, generation,
                     f"ERROR: camera {camera.name!r} not found. It may be "
                     "disconnected - reconnect the phone camera, press "
                     "Refresh, reselect it, then Start again."))
                return
            camera_index = resolved.index
            camera_backend = resolved.backend
            camera_name = resolved.display_name
        else:
            camera_index = int(camera)
        worker = CameraWorker(
            camera_index=camera_index,
            model_path=model_path,
            frame_queue=self.frame_queue,
            result_queue=self.result_queue,
            # Generation-stamped: messages from a superseded worker are
            # dropped by the tick, never applied to a new session.
            status=lambda m, g=generation: self.status_queue.put(
                (_STATUS_WORKER_MSG, g, m)),
            camera_backend=camera_backend,
            camera_name=camera_name)
        self.status_queue.put((_STATUS_WORKER_READY, generation, worker))

    def _on_worker_ready(self, generation: int, worker: CameraWorker) -> None:
        """Main thread only: adopt and start the prepared worker iff this
        preparation is still the current generation."""
        self._preparing = False
        if generation != self._generation or (
                self._prep_stop is not None and self._prep_stop.is_set()):
            return  # superseded/cancelled: camera never starts
        self.worker = worker
        worker.start()
        self._refresh_armed_status()

    def stop_tracking(self) -> None:
        """Stop capture. Generation is cancelled and the worker stopped
        FIRST, so even a failing input release cannot leave the camera or
        a preparation running; cleanup errors are collected and reported."""
        self._generation += 1  # any in-flight preparation becomes stale
        if self._prep_stop is not None:
            self._prep_stop.set()
        self._preparing = False
        self.sequencer.cancel()
        worker = self.worker
        self.worker = None
        if worker is not None:
            try:
                worker.stop()
            except Exception as exc:
                self._log_action(f"worker stop FAILED: {exc}")
        self._clear_queues()
        self._last_capture_at = None
        self._latest_hands = {}
        errors = []
        try:
            self.dispatcher.reset()
        except Exception as exc:
            errors.append(f"dispatcher reset: {exc}")
        try:
            self.controller.release_all("tracking stopped")
        except Exception as exc:
            errors.append(f"release: {exc}")
        if self.controller.armed:
            try:
                self.controller.set_armed(False)
            except Exception as exc:
                self.controller.armed = False  # disarm must survive cleanup
                errors.append(f"disarm: {exc}")
        self._focus_achieved = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self._set_status("idle", "Idle - tracking stopped")
        for err in errors:
            self._log_action(f"stop cleanup error ({err})")

    def _on_practice(self) -> None:
        # Mode switch (either direction): disarm and cancel any countdown
        # FIRST — a practice-armed state must never become live-armed
        # directly, and vice versa.
        self.sequencer.cancel()
        if self.controller.armed:
            self.controller.set_armed(False)
        self.controller.set_practice(self.practice_var.get())
        self.dispatcher.reset()  # practice state never carries into live
        self._focus_achieved = False
        self._refresh_armed_status()

    def _on_arm(self) -> None:
        if self.practice_var.get():
            self.dispatcher.reset()
            self.controller.set_armed(True)
            self._refresh_armed_status()
            return
        # Live arming: disarm first, then require a manually synced KNOWN
        # inventory belief (Mark open/closed) — never arm live while the
        # belief is UNKNOWN. The known belief is preserved across the reset.
        if self.controller.armed:
            self.controller.set_armed(False)
        if self.engine.inventory_believed is None:
            self.sequencer.cancel()
            self._log_action("live arm refused: inventory belief UNKNOWN - "
                             "press Mark open or Mark closed first")
            self._set_status(
                "alert",
                "Cannot ARM live: inventory belief UNKNOWN. "
                "Sync it with Mark open / Mark closed, then Arm again.")
            return
        self.dispatcher.reset(preserve_inventory=True)
        # 3-second countdown to switch to Minecraft. Output stays gated on
        # real Minecraft focus at dispatch time.
        self.sequencer.request(time.monotonic())
        self._focus_achieved = False
        self._set_status(
            "practice",
            f"Arming in {self.sequencer.countdown_s:.0f}s - "
            "switch to the Minecraft window now.")

    def _gestures_live_enabled(self) -> bool:
        """Gate for the result dispatcher: may gesture results advance the
        right-hand game state and fire binding actions right now?

        Practice mode never injects, so it is always allowed. Live mode is
        allowed only while armed, with Minecraft focused, and with a KNOWN
        inventory belief — a disarmed or unfocused live session must not
        let gestures set the belief without E, and punches are unsafe
        while the belief is unknown.

        The focus leg consults the immediate focus check — the callable the
        controller dispatches through, wired to ``minecraft_java_focused``
        — on EVERY call, never the 250 ms ``_focus_ok`` cache. If the check
        fails after focus was achieved, the gate fails closed itself:
        disarm, cancel any countdown, reset transient gesture state, mark
        the inventory belief UNKNOWN and clear both focus flags. This must
        happen here, not in ``_focus_tick``: a mid-dispatch disarm leaves
        the controller already disarmed when the tick next runs, and a
        stale known belief would then survive and allow re-arm without a
        manual sync. Before the first focus the session is just waiting:
        gate closed, no disarm."""
        if self.controller.practice:
            return True
        if not self.controller.armed:
            return False
        focus_check = getattr(self.controller, "_focus_check", None)
        if focus_check is None:
            focus_check = minecraft_java_focused
        try:
            focused = bool(focus_check())
        except Exception:
            focused = False
        self._focus_ok = focused
        if not focused:
            if not self._focus_achieved:
                return False  # armed but still waiting for the first focus
            self.controller.set_armed(False)
            self.sequencer.cancel()
            self.dispatcher.reset()  # transient state + inventory UNKNOWN
            self._focus_achieved = False
            return False
        return self.engine.inventory_believed is not None

    def _on_disarm(self) -> None:
        self.sequencer.cancel()
        self.controller.set_armed(False)
        self._focus_achieved = False
        self._refresh_armed_status()

    def _on_sensitivity(self, _value: str) -> None:
        self.engine.set_sensitivity(self.sens_var.get())
        if hasattr(self.dispatcher, "pose_engine"):
            self.dispatcher.pose_engine.set_sensitivity(self.sens_var.get())

    def _on_inventory_cursor_speed(self, _value: str) -> None:
        """Push the inventory cursor speed (px/sec) to the pose engine.

        Independent from the legacy Sensitivity slider: it never touches
        ``set_sensitivity``. Every hop is getattr-guarded because tk.Scale
        may invoke its command during early construction, before the
        dispatcher (or its pose engine) exists, and bare test shells use a
        dispatcher without a pose engine at all."""
        speed_var = getattr(self, "inventory_cursor_speed_var", None)
        dispatcher = getattr(self, "dispatcher", None)
        pose_engine = getattr(dispatcher, "pose_engine", None)
        setter = getattr(pose_engine, "set_inventory_cursor_speed", None)
        if speed_var is None or setter is None:
            return
        setter(speed_var.get())

    def _open_guide(self) -> None:
        from .guide import GestureGuide
        if self._guide is None or not self._guide.winfo_exists():
            self._guide = GestureGuide(self.root)
        else:
            self._guide.show()

    def _calibrate_poses(self) -> None:
        if not self._latest_hands:
            self._set_status("alert", "Show your hands to the camera before calibrating.")
            return
        self.controller.set_armed(False)
        self.sequencer.cancel()
        self.dispatcher.reset(preserve_inventory=True)
        self.dispatcher.pose_engine.calibrate(self._latest_hands)
        self._focus_achieved = self._focus_ok = False
        self._log_action("Neutral positions calibrated. Sync inventory and Arm to play.")
        self._refresh_armed_status()

    def _sync_inventory(self, is_open: bool) -> None:
        self.engine.set_inventory_state(is_open)
        self._log_action(f"manual inventory sync: {'open' if is_open else 'closed'}")

    def _refresh_armed_status(self) -> None:
        if self.sequencer.counting:
            return  # countdown status owns the label
        if self.controller.armed:
            if self.controller.practice:
                self._set_status("practice", "ARMED (practice - no real inputs)")
            elif self._focus_achieved:
                self._set_status("armed", "ARMED - live inputs to Minecraft")
            else:
                self._set_status(
                    "practice",
                    "ARMED - waiting for Minecraft focus "
                    "(no inputs until the game window is focused)")
        elif self.worker is not None:
            self._set_status("practice", "Tracking - disarmed")
        else:
            self._set_status("idle", "Idle - press Start to begin tracking")

    # --------------------------------------------------------------- ticks

    def _tick(self) -> None:
        """Main engine tick: drain queues, run engine, apply events."""
        try:
            # Pending releases/focus checks first: input servicing must not
            # wait behind result dispatch or preview cosmetics.
            self._service_inputs()
            self._tick_body()
        except Exception as exc:
            # Fail closed FIRST: disarm and cancel the countdown before any
            # best-effort cleanup, so a cleanup failure can never leave the
            # controller armed.
            try:
                self.controller.set_armed(False)
            except Exception:
                self.controller.armed = False
            self.sequencer.cancel()
            try:
                self.controller.release_all("error")
            except Exception as cleanup_exc:
                self._log_action(f"error cleanup FAILED: {cleanup_exc}")
            self._set_status("alert", f"ERROR: {exc}")
        finally:
            self.root.after(TICK_MS, self._tick)

    def _service_inputs(self) -> None:
        live_armed = self.controller.armed and not self.controller.practice
        try:
            self.controller.service_inputs()
        finally:
            if live_armed and not self.controller.armed:
                # A pending jump can discover focus loss outside result
                # dispatch. Invalidate the inventory belief here too.
                self.sequencer.cancel()
                self._focus_achieved = False
                self._focus_ok = False
                self.dispatcher.reset()

    def _tick_body(self) -> None:
        # Status packets from worker/preparation threads.
        while True:
            try:
                msg = self.status_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(msg, tuple):
                kind = msg[0]
                if kind == _STATUS_WORKER_READY:
                    self._on_worker_ready(msg[1], msg[2])
                elif kind == _STATUS_PREP_CANCELLED:
                    if msg[1] == self._generation:
                        self._preparing = False
                        self.start_btn.configure(state="normal")
                        self.stop_btn.configure(state="disabled")
                        self._set_status("idle", "Preparation cancelled.")
                elif kind == _STATUS_PREP_FAILED:
                    if msg[1] == self._generation:
                        self._preparing = False
                        self.stop_tracking()
                        self._set_status("alert", msg[2])
                elif kind == _STATUS_WORKER_MSG:
                    if msg[1] != self._generation or self.worker is None:
                        continue  # stale worker: cannot touch this session
                    text = msg[2]
                    if text.startswith("ERROR"):
                        self.stop_tracking()
                        self._set_status("alert", text)
                    else:
                        self._set_status(
                            "practice" if not self.controller.armed else
                            ("armed" if not self.controller.practice
                             else "practice"),
                            text)
                continue
            # Plain strings are preparation progress messages; only the
            # current preparation may show them.
            if not self._preparing:
                continue
            self._set_status("practice", msg)

        now = time.monotonic()  # fresh timestamp for dispatch, taken
        # immediately after status work and BEFORE any expensive cosmetics

        # Arming countdown completion.
        if self.sequencer.poll(now):
            self.controller.set_armed(True)
            self._refresh_armed_status()
        elif self.sequencer.counting:
            self._set_status(
                "practice",
                f"Arming in {self.sequencer.remaining(now):.0f}s - "
                "switch to the Minecraft window now.")

        # Tracking results: drain the queue but act on the NEWEST packet
        # only — older queued packets are dropped unprocessed. Packets are
        # only accepted while a worker of the current generation is active.
        # This runs BEFORE the camera preview decode so input recognition
        # and input release are never delayed by cosmetic work.
        packet = None
        while True:
            try:
                packet = self.result_queue.get_nowait()
            except queue.Empty:
                break

        got_result = False
        if packet is not None and self.worker is not None:
            hands, captured_at = packet
            # Packets older than 300 ms of total capture age are stale:
            # rejecting one releases any held input via tracking-loss.
            if self.dispatcher.dispatch(hands, captured_at, now):
                got_result = True
                self._last_capture_at = captured_at
                self._latest_hands = hands or {}
            else:
                self.dispatcher.tracking_lost(now)
                self._latest_hands = {}

        # Stale stream: >300 ms of total capture age without a fresh packet
        # while tracking -> treat as tracking loss so held inputs release.
        if (self.worker is not None and not got_result
                and self._last_capture_at is not None
                and now - self._last_capture_at > STALE_FRAME_S):
            self.dispatcher.tracking_lost(now)
            self._latest_hands = {}
            self._last_capture_at = now

        # Camera preview (latest frame wins; drop the rest). Purely
        # cosmetic: a decode/display failure must never block or tear down
        # recognition and input release, so it is contained here.
        ppm = None
        while True:
            try:
                ppm = self.frame_queue.get_nowait()
            except queue.Empty:
                break
        if ppm is not None and self.worker is not None:
            try:
                self._photo = tk.PhotoImage(data=ppm, format="PPM")
                self.preview.configure(image=self._photo, text="")
            except Exception as exc:
                self._log_action(f"preview update failed (ignored): {exc}")

        self._update_config_panel()
        if getattr(self, "_guide", None) is not None:
            pose = getattr(self.dispatcher, "last_pose", None)
            labels = pose.labels if pose is not None else {"left": "Neutral", "right": "Neutral"}
            self._guide.update_state(labels, self.engine.inventory_believed is True)

        believed = self.engine.inventory_believed
        text = {True: "open", False: "closed", None: "UNKNOWN"}[believed]
        self.inv_label.configure(text=f"Inventory belief: {text}")
        self._update_control_overlay()

    def _update_control_overlay(self) -> None:
        """Hand the tracker worker an immutable snapshot of control state
        for its overlay drawing. Runs at the end of each tick, after the
        latest result packet has been processed. No mutable engine object
        is shared: the worker gets plain tuples/dicts in one assignment."""
        worker = self.worker
        if worker is None:
            return
        pe = getattr(self.dispatcher, "pose_engine", None)
        if pe is None:
            return
        pose = getattr(self.dispatcher, "last_pose", None)
        left_locked = getattr(pe, "_left_center", None)
        right_locked = getattr(pe, "_right_center", None)
        right_center = right_locked or getattr(pe, "_right_anchor", None)
        if right_center is not None:
            right_center = tuple(right_center)
        cursor_locked = getattr(pe, "_cursor_center", None)
        cursor_center = cursor_locked or getattr(pe, "_cursor_anchor", None)
        if cursor_center is not None:
            cursor_center = tuple(cursor_center)
        if pose is not None:
            held_keys = tuple(getattr(pose, "held_keys", None) or ())
            labels = dict(getattr(pose, "labels", None) or {})
            hotbar_active = bool(getattr(pose, "hotbar_active", None) or False)
        else:
            held_keys = ()
            labels = {}
            hotbar_active = False
        snapshot = {
            "left_center": left_locked or getattr(pe, "_left_anchor", None),
            "right_center": right_center,
            "cursor_center": cursor_center,
            "held_keys": held_keys,
            "labels": labels,
            "inventory_open": self.engine.inventory_believed is True,
            "hotbar_active": hotbar_active,
            "left_ready": left_locked is not None,
            "right_ready": right_locked is not None,
            "cursor_ready": cursor_locked is not None,
        }
        worker.control_overlay_state = snapshot

    def _focus_tick(self) -> None:
        try:
            focused = minecraft_java_focused()
            self._focus_ok = focused
            live_armed = self.controller.armed and not self.controller.practice
            if live_armed and focused and not self._focus_achieved:
                self._focus_achieved = True
                self._refresh_armed_status()
            elif live_armed and not focused and self._focus_achieved:
                # Fail closed: release, disarm, demand explicit re-arm.
                # After an uncertain focus loss the inventory belief is
                # UNKNOWN until the user syncs it manually. Only an ACTUAL
                # focus loss after focus was acquired disarms — a waiting
                # armed state (countdown done, game not yet focused) keeps
                # waiting, with output blocked by the dispatcher gate.
                self.controller.release_all("focus lost")
                self.controller.set_armed(False)
                self.sequencer.cancel()
                self._focus_achieved = False
                self.engine.set_inventory_state(None)
                self._set_status("alert",
                                 "FOCUS LOST - inputs released, disarmed, "
                                 "inventory UNKNOWN. Refocus Minecraft, sync "
                                 "inventory if needed, and press Arm to re-arm.")
        except Exception:
            self._focus_ok = False
            if (self.controller.armed and not self.controller.practice
                    and self._focus_achieved):
                self.controller.release_all("focus check error")
                self.controller.set_armed(False)
                self.sequencer.cancel()
                self._focus_achieved = False
                self.engine.set_inventory_state(None)
                self._set_status("alert", "Focus check failed - disarmed.")
        finally:
            self.root.after(FOCUS_CHECK_MS, self._focus_tick)

    def _f8_tick(self) -> None:
        """Global F8 emergency stop, independent of window focus. Works
        during model preparation too: it cancels the download."""
        try:
            down = emergency_stop_pressed()
            if down and not self._f8_was_down:
                self._emergency_stop()
            self._f8_was_down = down
        finally:
            self.root.after(F8_CHECK_MS, self._f8_tick)

    def _emergency_stop(self) -> None:
        # Best-effort release first; even if it raises, the generation is
        # cancelled and the worker stopped by stop_tracking below.
        try:
            self.controller.release_all("EMERGENCY STOP")
            self.controller.set_armed(False)
        except Exception as exc:
            self.controller.armed = False
            self._log_action(f"emergency release FAILED: {exc}")
        try:
            self.stop_tracking()
        except Exception as exc:
            # Last-resort: cancel generation and stop the worker directly.
            self._log_action(f"emergency stop cleanup FAILED: {exc}")
            self._generation += 1
            if self._prep_stop is not None:
                self._prep_stop.set()
            worker = self.worker
            self.worker = None
            if worker is not None:
                try:
                    worker.stop()
                except Exception:
                    pass
        self._set_status("alert", "EMERGENCY STOP (F8) - everything halted.")

    # -------------------------------------------------------------- shutdown

    def _on_close(self) -> None:
        try:
            self._close_config_panel()
            self.stop_tracking()
            self.controller.release_all("exit")
        finally:
            self.root.destroy()


def main() -> None:
    root = tk.Tk()
    root.minsize(900, 560)
    app = HandCraftApp(root)
    if getattr(app, "_defer_log", None):
        app._log_action(app._defer_log)
    root.mainloop()


if __name__ == "__main__":
    main()
