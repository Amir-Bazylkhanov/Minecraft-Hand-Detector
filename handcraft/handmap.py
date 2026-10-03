"""Tkinter hand reference and gesture-binding configuration panel.

The panel is a development/configuration aid for the Minecraft camera-gesture
app.  It displays numbered left and right hand maps, the palm-centroid
reference, and editable gesture-to-action assignments.  Configuration is saved
as JSON in the current workspace (``handcraft_bindings.json`` by default).

Integration notes
-----------------
Run ``python -m handcraft.handmap`` from the workspace to open the panel
standalone, or embed ``GestureConfigPanel`` in an existing Tk app::

    GestureConfigPanel(parent=None, config_path=None, on_bindings_changed=None)

The panel is a ``tk.Toplevel``.  With ``parent=None`` it creates and owns one
hidden ``tk.Tk`` root, which is destroyed when the panel closes.  With a
parent supplied it attaches to that parent and never creates another root.
After a successful Save the panel rebuilds its recognizer from the saved
bindings and invokes ``on_bindings_changed(bindings)`` with a bindings dict
accepted directly by ``PalmContactRecognizer(bindings=...)`` from
``handcraft.bindings``.

This module never opens a camera, captures global input, sends keystrokes, or
controls Minecraft.  Its only external side effect is saving/loading the JSON
configuration file requested by the user.  ``draw_landmark_overlay`` is an
optional OpenCV convenience for an already-created video frame; cv2 is imported
lazily only when that helper is called.
"""

from __future__ import annotations

import copy
import json
from math import isfinite
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import tkinter as tk
from tkinter import messagebox, ttk

try:  # Package import when used as python -m handcraft.handmap.
    from .adapters import vk_for_key
    from .bindings import DEFAULT_BINDINGS, PALM_POINTS, PalmContactRecognizer
except ImportError:  # Direct script execution from the handcraft directory.
    from adapters import vk_for_key  # type: ignore
    from bindings import DEFAULT_BINDINGS, PALM_POINTS, PalmContactRecognizer  # type: ignore

CONFIG_FILE_NAME = "handcraft_bindings.json"
CONFIG_VERSION = 1

LANDMARK_REFERENCE = (
    (0, "wrist", "Wrist"),
    (4, "thumb", "Thumb tip"),
    (8, "index", "Index tip"),
    (12, "middle", "Middle tip"),
    (16, "ring", "Ring tip"),
    (20, "pinky", "Pinky tip"),
)

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

# Normalized illustration coordinates.  Y increases downward for Tkinter.
_REFERENCE_POINTS = {
    0: (0.50, 0.91),
    1: (0.31, 0.74), 2: (0.20, 0.61), 3: (0.11, 0.49), 4: (0.05, 0.39),
    5: (0.35, 0.59), 6: (0.29, 0.42), 7: (0.27, 0.30), 8: (0.27, 0.18),
    9: (0.48, 0.56), 10: (0.48, 0.37), 11: (0.48, 0.24), 12: (0.48, 0.10),
    13: (0.61, 0.59), 14: (0.66, 0.42), 15: (0.69, 0.30), 16: (0.71, 0.19),
    17: (0.73, 0.65), 18: (0.82, 0.53), 19: (0.88, 0.44), 20: (0.94, 0.35),
}


def default_config_path() -> Path:
    """Return the JSON binding file used by the standalone workspace panel."""

    return Path.cwd() / CONFIG_FILE_NAME


def _default_config() -> Dict[str, Any]:
    return {
        "version": CONFIG_VERSION,
        "bindings": copy.deepcopy(DEFAULT_BINDINGS),
    }


def load_binding_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load panel JSON, returning defaults when the workspace file is absent."""

    config_path = Path(path) if path is not None else default_config_path()
    if not config_path.exists():
        return _default_config()
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    bindings = data.get("bindings")
    if not isinstance(bindings, dict):
        raise ValueError(f"{config_path} does not contain a JSON 'bindings' object")
    merged = _default_config()
    merged["version"] = int(data.get("version", CONFIG_VERSION))
    merged["bindings"].update(copy.deepcopy(bindings))
    return merged


def save_binding_config(config: Mapping[str, Any], path: Optional[Path] = None) -> Path:
    """Persist editable bindings to the workspace JSON file and return its path."""

    config_path = Path(path) if path is not None else default_config_path()
    payload = {
        "version": int(config.get("version", CONFIG_VERSION)),
        "bindings": copy.deepcopy(dict(config.get("bindings", {}))),
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return config_path


def command_text(bindings: Optional[Mapping[str, Mapping[str, Any]]] = None) -> str:
    """Return human-readable English hand-map and command text for clipboard use."""

    active = copy.deepcopy(DEFAULT_BINDINGS if bindings is None else bindings)
    lines = [
        "Minecraft hand-gesture reference",
        "",
        "Left hand and right hand landmark map:",
    ]
    for number, finger, label in LANDMARK_REFERENCE:
        lines.append(f"  {number}: {label} ({finger})")
    palm_list = ", ".join(str(point) for point in PALM_POINTS)
    lines.extend([
        f"  PALM: centroid of landmarks {palm_list}",
        "",
        "Configured gesture commands:",
    ])
    for gesture_id in sorted(active):
        binding = active[gesture_id]
        hand = str(binding.get("hand", "unknown")).title()
        finger = str(binding.get("finger", "unknown"))
        action = str(binding.get("action", "hold")).lower()
        key = str(binding.get("key", "")).upper()
        description = str(binding.get("description", "")).strip()
        verb = "Hold" if action == "hold" else "Pulse"
        lines.append(f"  {gesture_id}: {verb} {key} — {hand} {finger} tip touches palm.")
        if description:
            lines.append(f"    {description}")
    lines.extend([
        "",
        "Recognition uses palm-normalized distances with enter/exit hysteresis.",
        "Tracking loss releases held keys; pulse gestures fire once per contact.",
    ])
    return "\n".join(lines)


def draw_landmark_overlay(
    frame: Any,
    hands: Mapping[str, Sequence[Any]],
    *,
    left_color: Tuple[int, int, int] = (40, 220, 40),
    right_color: Tuple[int, int, int] = (60, 160, 255),
    palm_color: Tuple[int, int, int] = (0, 220, 255),
    draw_palm: bool = True,
) -> Any:
    """Draw landmarks on an existing OpenCV BGR frame and return that frame.

    ``hands`` uses the same handedness-to-landmarks contract as
    ``PalmContactRecognizer.update_hands``.  Normalized coordinates (0..1) are
    scaled to frame width/height; pixel coordinates are used as-is.  OpenCV is
    optional and imported only inside this helper.  A clear ``ImportError`` is
    raised when cv2 is unavailable.
    """

    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "draw_landmark_overlay requires the optional 'cv2' package; "
            "recognition and tests do not require it"
        ) from exc

    height, width = frame.shape[:2]
    for handedness, landmarks in hands.items():
        label = str(handedness).strip().lower()
        if label not in {"left", "right"} or len(landmarks) < 21:
            continue
        points = []
        valid = True
        for landmark in landmarks:
            try:
                x, y = _point_xy(landmark)
            except (TypeError, ValueError, KeyError, IndexError):
                valid = False
                break
            px = int(round(x * (width - 1))) if 0.0 <= x <= 1.0 else int(round(x))
            py = int(round(y * (height - 1))) if 0.0 <= y <= 1.0 else int(round(y))
            points.append((px, py))
        if not valid:
            continue
        color = left_color if label == "left" else right_color
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], color, 2, cv2.LINE_AA)
        for index, point in enumerate(points):
            radius = 5 if index in {0, 4, 8, 12, 16, 20} else 3
            cv2.circle(frame, point, radius, color, -1, cv2.LINE_AA)
            if index in {0, 4, 8, 12, 16, 20}:
                cv2.putText(
                    frame,
                    str(index),
                    (point[0] + 5, point[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        if draw_palm:
            palm_x = sum(points[index][0] for index in PALM_POINTS) // len(PALM_POINTS)
            palm_y = sum(points[index][1] for index in PALM_POINTS) // len(PALM_POINTS)
            cv2.circle(frame, (palm_x, palm_y), 6, palm_color, -1, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"{label.title()} PALM",
                (palm_x + 8, palm_y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                palm_color,
                1,
                cv2.LINE_AA,
            )
    return frame


def _point_xy(point: Any) -> Tuple[float, float]:
    if hasattr(point, "x") and hasattr(point, "y"):
        return float(point.x), float(point.y)
    if isinstance(point, Mapping):
        return float(point["x"]), float(point["y"])
    return float(point[0]), float(point[1])


class GestureConfigPanel(tk.Toplevel):
    """Tk reference map and editable binding panel.

    The panel edits only action and key assignment fields.  It deliberately has
    no camera preview and no key-sending controls.

    With ``parent=None`` the panel owns one hidden ``tk.Tk`` root and destroys
    it when the panel closes; with a parent supplied it attaches to that
    parent's root.  ``on_bindings_changed`` is invoked with the bindings dict
    after every successful Save.
    """

    def __init__(
        self,
        parent: Optional[tk.Misc] = None,
        config_path: Optional[Path] = None,
        on_bindings_changed: Optional[Callable[[Dict[str, Dict[str, Any]]], None]] = None,
    ) -> None:
        self._owned_root: Optional[tk.Tk] = None
        if parent is None:
            self._owned_root = tk.Tk()
            self._owned_root.withdraw()
            master: tk.Misc = self._owned_root
        else:
            master = parent
        super().__init__(master)
        self._on_bindings_changed = on_bindings_changed
        self.title("Minecraft Hand Gesture Bindings")
        self.minsize(900, 640)
        self.config_path = Path(config_path) if config_path is not None else default_config_path()
        self.config_data = load_binding_config(self.config_path)
        self._editors: Dict[str, Tuple[ttk.Combobox, tk.Entry]] = {}
        self._live_recognizer: Optional[PalmContactRecognizer] = None

        self._build_reference_canvas()
        self._build_editor()
        self._build_command_text()
        self._build_buttons()

        if self._owned_root is not None:
            self.protocol("WM_DELETE_WINDOW", self._close_standalone)

    def _close_standalone(self) -> None:
        self.destroy()
        if self._owned_root is not None:
            self._owned_root.destroy()
            self._owned_root = None

    def _build_reference_canvas(self) -> None:
        frame = ttk.LabelFrame(self, text="Numbered left/right hand reference")
        frame.pack(fill="x", padx=10, pady=(10, 6))
        canvas = tk.Canvas(frame, width=860, height=255, bg="#17202a", highlightthickness=0)
        canvas.pack(fill="x", padx=6, pady=6)
        self._hand_canvas = canvas
        # hand label -> (x, y, width, height, mirror) viewport within the canvas.
        self._hand_viewports = {
            "left": (30, 14, 390, 225, False),
            "right": (440, 14, 390, 225, True),
        }
        self._tracking_var = tk.StringVar(value="Hand tracking: no live data")
        ttk.Label(frame, textvariable=self._tracking_var, font=("Segoe UI", 9)).pack(
            anchor="w", padx=8, pady=(0, 6)
        )
        self._redraw_hands({})

    def _redraw_hands(
        self,
        live_points: Mapping[str, Optional[Sequence[Tuple[float, float]]]],
    ) -> None:
        canvas = self._hand_canvas
        canvas.delete("all")
        for hand, (x, y, width, height, mirror) in self._hand_viewports.items():
            points = live_points.get(hand)
            self._draw_hand(canvas, x, y, width, height, hand.upper(), mirror, points)
        palm_points = ", ".join(str(point) for point in PALM_POINTS)
        canvas.create_text(
            430,
            242,
            text=f"PALM centroid = average of landmarks {palm_points}",
            fill="#f7dc6f",
            font=("Segoe UI", 9, "bold"),
        )

    def update_hands(
        self,
        hands: Mapping[str, Sequence[Any]],
        contacts: Any = None,
    ) -> None:
        """Redraw the hand canvases from live landmarks and show contact state.

        Call from the Tk GUI thread once per camera frame.  ``hands`` maps a
        handedness label (``"Left"``/``"Right"``, case-insensitive) to at
        least 21 landmarks; each landmark may be a ``(x, y)`` tuple/list, a
        mapping with ``"x"``/``"y"`` entries, or an object with ``x``/``y``
        attributes.  Live points are bounding-box scaled into each hand
        viewport; hands that are missing or malformed keep the numbered
        reference diagram.

        ``contacts`` may be a gesture-id-to-bool mapping, a
        ``RecognitionResult`` (uses its ``contact_states``), or a
        ``PalmContactRecognizer`` (queried with this frame).  When omitted, an
        internal recognizer with the panel's current bindings is used.  This
        method never opens a camera or sends input.
        """

        live: Dict[str, Optional[Sequence[Tuple[float, float]]]] = {}
        tracked: Dict[str, bool] = {"left": False, "right": False}
        if isinstance(hands, Mapping):
            for handedness, landmarks in hands.items():
                label = str(handedness).strip().lower()
                if label not in self._hand_viewports:
                    continue
                points = self._live_points(landmarks)
                if points is not None:
                    live[label] = points
                    tracked[label] = True

        states = self._contact_states(hands, contacts)
        self._redraw_hands(live)

        parts = [f"{hand.title()}: {'tracked' if tracked[hand] else 'lost'}" for hand in ("left", "right")]
        lines = ["Hand tracking: " + " | ".join(parts)]
        if states:
            active = [gid for gid, on in states.items() if on]
            if active:
                lines.append("Contact: " + ", ".join(sorted(active)))
            else:
                lines.append("Contact: none")
        self._tracking_var.set("\n".join(lines))

    def _live_points(self, landmarks: Sequence[Any]) -> Optional[Tuple[Tuple[float, float], ...]]:
        try:
            if len(landmarks) < 21:
                return None
            points = tuple(_point_xy(landmark) for landmark in landmarks)
        except (TypeError, ValueError, KeyError, IndexError, AttributeError):
            return None
        if any(not (isfinite(x) and isfinite(y)) for x, y in points):
            return None
        return points

    def _contact_states(
        self,
        hands: Mapping[str, Sequence[Any]],
        contacts: Any,
    ) -> Dict[str, bool]:
        if contacts is None:
            if self._live_recognizer is None:
                self._live_recognizer = PalmContactRecognizer(bindings=self._current_bindings())
            return dict(self._live_recognizer.update_hands(hands).contact_states)
        if isinstance(contacts, PalmContactRecognizer):
            return dict(contacts.update_hands(hands).contact_states)
        if hasattr(contacts, "contact_states"):
            return dict(contacts.contact_states)
        if isinstance(contacts, Mapping):
            return {str(gid): bool(on) for gid, on in contacts.items()}
        return {}

    def _draw_hand(
        self,
        canvas: tk.Canvas,
        x: int,
        y: int,
        width: int,
        height: int,
        label: str,
        mirror: bool,
        live: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> None:
        live_map: Optional[Dict[int, Tuple[float, float]]] = None
        if live is not None and len(live) >= 21:
            xs = [point[0] for point in live]
            ys = [point[1] for point in live]
            span_x = max(xs) - min(xs)
            span_y = max(ys) - min(ys)
            margin = 0.08
            usable_w = width * (1.0 - 2 * margin)
            usable_h = height * (1.0 - 2 * margin)
            scale = min(
                usable_w / span_x if span_x > 1e-9 else usable_h,
                usable_h / span_y if span_y > 1e-9 else usable_w,
            )
            offset_x = x + margin * width - min(xs) * scale
            offset_y = y + margin * height - min(ys) * scale
            live_map = {
                index: (offset_x + live[index][0] * scale, offset_y + live[index][1] * scale)
                for index in range(21)
            }

        def mapped(index: int) -> Tuple[float, float]:
            if live_map is not None:
                return live_map[index]
            nx, ny = _REFERENCE_POINTS[index]
            if mirror:
                nx = 1.0 - nx
            return x + nx * width, y + ny * height

        caption = label if live_map is None else f"{label} (live)"
        canvas.create_text(
            x + width / 2,
            y + 4,
            text=caption,
            fill="white",
            font=("Segoe UI", 11, "bold"),
        )
        for start, end in HAND_CONNECTIONS:
            canvas.create_line(*mapped(start), *mapped(end), fill="#5dade2", width=3)
        for index in range(21):
            px, py = mapped(index)
            major = index in {0, 4, 8, 12, 16, 20}
            radius = 7 if major else 4
            color = "#f8f9f9" if major else "#85c1e9"
            canvas.create_oval(px - radius, py - radius, px + radius, py + radius, fill=color, outline="")
            if major:
                name = next(name for number, name, _ in LANDMARK_REFERENCE if number == index)
                canvas.create_text(
                    px,
                    py - 14,
                    text=f"{index} {name}",
                    fill="#f7dc6f",
                    font=("Segoe UI", 8, "bold"),
                )
        palm_x = sum(mapped(index)[0] for index in PALM_POINTS) / len(PALM_POINTS)
        palm_y = sum(mapped(index)[1] for index in PALM_POINTS) / len(PALM_POINTS)
        canvas.create_oval(palm_x - 8, palm_y - 8, palm_x + 8, palm_y + 8, fill="#f7dc6f", outline="")
        canvas.create_text(
            palm_x,
            palm_y + 18,
            text="PALM",
            fill="#f7dc6f",
            font=("Segoe UI", 8, "bold"),
        )

    def _build_editor(self) -> None:
        frame = ttk.LabelFrame(self, text="Editable gesture-to-action assignments")
        frame.pack(fill="x", padx=10, pady=6)
        headings = ("Gesture", "Meaning", "Action", "Key")
        for column, heading in enumerate(headings):
            ttk.Label(frame, text=heading, font=("Segoe UI", 9, "bold")).grid(
                row=0, column=column, sticky="w", padx=5, pady=(5, 2)
            )
        for row, gesture_id in enumerate(sorted(self.config_data["bindings"]), start=1):
            binding = self.config_data["bindings"][gesture_id]
            ttk.Label(frame, text=gesture_id).grid(row=row, column=0, sticky="w", padx=5)
            meaning = f"{binding.get('hand', '?').title()} {binding.get('finger', '?')} tip to palm"
            ttk.Label(frame, text=meaning).grid(row=row, column=1, sticky="w", padx=5)
            action = ttk.Combobox(
                frame,
                values=("hold", "pulse"),
                width=8,
                state="readonly",
            )
            action.set(str(binding.get("action", "hold")))
            action.grid(row=row, column=2, sticky="w", padx=5, pady=2)
            key = tk.Entry(frame, width=14)
            key.insert(0, str(binding.get("key", "")))
            key.grid(row=row, column=3, sticky="w", padx=5, pady=2)
            self._editors[gesture_id] = (action, key)
        frame.columnconfigure(1, weight=1)

    def _build_command_text(self) -> None:
        frame = ttk.LabelFrame(self, text="Descriptive command text")
        frame.pack(fill="both", expand=True, padx=10, pady=6)
        self.command_box = tk.Text(frame, height=10, wrap="word", font=("Consolas", 9))
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.command_box.yview)
        self.command_box.configure(yscrollcommand=scrollbar.set)
        self.command_box.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)
        scrollbar.pack(side="right", fill="y", padx=(0, 6), pady=6)
        self._refresh_command_text()

    def _build_buttons(self) -> None:
        frame = ttk.Frame(self)
        frame.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(frame, text="Copy command text", command=self.copy_command_text).pack(side="left")
        ttk.Button(frame, text="Refresh", command=self._refresh_command_text).pack(side="left", padx=6)
        ttk.Button(frame, text="Save JSON", command=self.save).pack(side="right")
        ttk.Label(frame, text=f"Workspace file: {self.config_path}").pack(side="right", padx=8)

    def _current_bindings(self) -> Dict[str, Dict[str, Any]]:
        bindings = copy.deepcopy(self.config_data["bindings"])
        for gesture_id, (action_editor, key_editor) in self._editors.items():
            bindings[gesture_id]["action"] = action_editor.get().strip().lower()
            bindings[gesture_id]["key"] = key_editor.get().strip().upper()
        return bindings

    def _refresh_command_text(self) -> None:
        self.command_box.delete("1.0", "end")
        self.command_box.insert("1.0", command_text(self._current_bindings()))

    def copy_command_text(self) -> None:
        """Copy the descriptive English command text to the Windows clipboard."""

        text = self.command_box.get("1.0", "end-1c")
        self.clipboard_clear()
        self.clipboard_append(text)
        self.update()  # Keep clipboard content available after focus changes.

    def save(self) -> None:
        """Validate and save edited assignments to the workspace JSON file.

        Validation runs *before* any state change: the edited bindings must
        be accepted by ``PalmContactRecognizer`` and every key must map to a
        virtual-key code via ``vk_for_key``.  Malformed values (missing or
        wrongly-typed fields) are rejected the same way.  On any failure the
        panel keeps its previous in-memory config, the JSON file is left
        untouched, the live recognizer is unchanged, and
        ``on_bindings_changed`` is not invoked.
        """

        try:
            bindings = self._current_bindings()
        except (TypeError, KeyError, AttributeError) as exc:
            messagebox.showerror(
                "Invalid binding",
                f"Edited bindings are malformed: {exc}",
            )
            return
        for gesture_id, binding in bindings.items():
            if not isinstance(binding, dict):
                messagebox.showerror(
                    "Invalid binding",
                    f"{gesture_id}: binding must be a mapping of fields",
                )
                return
            action = str(binding.get("action", "")).strip().lower()
            key = binding.get("key")
            if action not in {"hold", "pulse"}:
                messagebox.showerror("Invalid action", f"{gesture_id}: action must be hold or pulse")
                return
            if key is None or not str(key).strip():
                messagebox.showerror("Missing key", f"{gesture_id}: key must not be empty")
                return
            if vk_for_key(key) is None:
                messagebox.showerror(
                    "Unknown key",
                    f"{gesture_id}: {key!r} does not map to a "
                    "virtual-key code",
                )
                return
        try:
            recognizer = PalmContactRecognizer(bindings=copy.deepcopy(bindings))
        except Exception as exc:
            messagebox.showerror("Invalid binding", str(exc))
            return
        self.config_data["bindings"] = bindings
        saved_path = save_binding_config(self.config_data, self.config_path)
        self._live_recognizer = recognizer
        self._refresh_command_text()
        if self._on_bindings_changed is not None:
            self._on_bindings_changed(copy.deepcopy(bindings))
        messagebox.showinfo("Saved", f"Bindings saved to:\n{saved_path}")


def main() -> None:
    """Open the standalone reference/configuration panel and run its loop."""

    panel = GestureConfigPanel()
    panel.mainloop()


__all__ = [
    "CONFIG_FILE_NAME",
    "GestureConfigPanel",
    "HAND_CONNECTIONS",
    "LANDMARK_REFERENCE",
    "command_text",
    "default_config_path",
    "draw_landmark_overlay",
    "load_binding_config",
    "main",
    "save_binding_config",
]


if __name__ == "__main__":
    main()
