"""A compact, passive gesture reference attached to the application's Tk root."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from collections.abc import Mapping

# Each illustration describes the complete pose; green fingers are extended.
GESTURE_CARDS = (
    ("LEFT · move", {8}, "Move hand from center: ↑ forward · ↓ back · ← / → strafe.\nCenter stops; farther forward sprints."),
    ("LEFT · jump", {4}, "Thumb out alone or while pointing/V holds Space (keeps jumping).\nFold the thumb to release."),
    ("LEFT · sneak", {8, 12}, "V sign: use the same hand joystick while sneaking."),
    ("RIGHT · look", {8}, "Point and move hand from center to turn.\nIn inventory: the cursor keeps its entry spot; move the wrist from center."),
    ("RIGHT · mine / attack", {4}, "Thumb out-in = one attack; repeat within 0.6 s holds mining,\nrefreshing 0.8 s. Hand: fist or point; not V, open, or hotbar."),
    ("RIGHT · use / place", {8, 12}, "Hold a V sign briefly for one right-click.\nInventory click: thumb out-in instead. Change pose to repeat."),
    ("BOTH · inventory", {4, 8, 12, 16, 20}, "Show both open palms to toggle instantly.\nChange pose before opening or closing again."),
    ("RIGHT · hotbar", {4, 20}, "Start with thumb + pinky out (🤙). Fold thumb: ← slot.\nFold pinky: → slot. Takes priority over attack."),
)


class GestureGuide(tk.Toplevel):
    """Reference window; updates never raise the window or take keyboard focus."""

    def __init__(self, parent: tk.Misc):
        super().__init__(parent)
        self.title("Minecraft Hand Detector · gesture guide")
        self.geometry("340x430")
        self.minsize(340, 430)
        self.configure(background="#12202b")
        self.topmost = tk.BooleanVar(self, value=True)
        self.opacity = tk.DoubleVar(self, value=0.55)
        self.attributes("-topmost", True)
        self.attributes("-alpha", self.opacity.get())
        self.protocol("WM_DELETE_WINDOW", self.withdraw)
        header = ttk.Frame(self, padding=(8, 5))
        header.pack(fill="x")
        controls = ttk.Frame(header)
        controls.pack(fill="x")
        ttk.Checkbutton(controls, text="On top", variable=self.topmost,
                        command=self._set_topmost).pack(side="left")
        ttk.Label(controls, text="Opacity").pack(side="left", padx=(8, 4))
        ttk.Scale(controls, from_=0.25, to=1.0, variable=self.opacity,
                  command=self._set_opacity, length=105).pack(side="left")
        self.opacity_text = tk.StringVar(self, value="55%")
        ttk.Label(controls, textvariable=self.opacity_text, width=4).pack(side="left", padx=3)
        self.state_text = tk.StringVar(self, value="Waiting for tracking")
        ttk.Label(header, textvariable=self.state_text, wraplength=320,
                  font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(4, 0))
        tk.Label(self, text="Tips: thumb 4 · index 8 · middle 12\nring 16 · pinky 20 · P = palm",
                 background="#12202b", foreground="#b9cbd7", font=("Segoe UI", 8),
                 justify="left", anchor="w").pack(fill="x", padx=8, pady=(0, 4))
        compact_cards = (
            ("LEFT · move · 8 up", "Hand joystick; center stops, far forward sprints."),
            ("LEFT · jump · 4 up", "Thumb alone or with point/V holds Space; fold releases."),
            ("LEFT · sneak · 8 + 12 up", "V sign + hand joystick = sneak movement."),
            ("RIGHT · look · 8 up", "Move from center to turn; inventory: wrist moves cursor."),
            ("RIGHT · mine · thumb out-in", "Thumb out-in: attack; repeat: mine; open stops."),
            ("RIGHT · use · 8 + 12 up", "V: use. Inventory: point + thumb out-in clicks."),
            ("BOTH · inventory · open palms", "Both palms toggle instantly; change pose to rearm."),
            ("RIGHT · hotbar · 4 + 20 up", "Fold 4: ← slot; fold 20: → slot; overrides attack."),
        )
        for title, instruction in compact_cards:
            row = tk.Frame(self, background="#12202b")
            row.pack(fill="x", padx=8, pady=0)
            tk.Label(row, text=title, background="#12202b", foreground="#68dfb0",
                     font=("Segoe UI", 9, "bold"), anchor="w", pady=0).pack(fill="x")
            tk.Label(row, text=instruction, background="#12202b", foreground="#ecf3f8",
                     font=("Segoe UI", 8), anchor="w", pady=0).pack(fill="x")
        tk.Label(self, text="Open hand = neutral · F8 stops controls",
                 background="#12202b", foreground="#b9cbd7",
                 font=("Segoe UI", 8), anchor="w").pack(fill="x", padx=8, pady=(5, 3))

    def _set_topmost(self):
        self.attributes("-topmost", bool(self.topmost.get()))

    def _set_opacity(self, value):
        opacity = max(0.25, min(1.0, float(value)))
        self.attributes("-alpha", opacity)
        self.opacity_text.set(f"{opacity:.0%}")

    @staticmethod
    def _draw_pose(canvas: tk.Canvas, extended: set[int]):
        canvas.create_rectangle(31, 47, 88, 82, outline="#91a5b3", fill="#293c49", width=2)
        for tip, x, top in ((8, 38, 18), (12, 53, 11), (16, 68, 17), (20, 83, 27)):
            out = tip in extended
            end_y = top if out else 65
            canvas.create_line(x, 48, x, end_y, width=8, fill="#68dfb0" if out else "#7d8a93", capstyle="round")
            canvas.create_text(x, end_y - 9 if out else 76, text=str(tip), fill="#ecf3f8", font=("Segoe UI", 8))
        out = 4 in extended
        canvas.create_line(33, 67, 12 if out else 48, 46 if out else 61,
                           width=8, fill="#68dfb0" if out else "#7d8a93", capstyle="round")
        canvas.create_text(12 if out else 24, 33 if out else 78, text="4", fill="#ecf3f8", font=("Segoe UI", 8))
        canvas.create_text(58, 91, text="P · palm", fill="#91a5b3", font=("Segoe UI", 8))

    def update_state(self, labels: Mapping | str, inventory_open: bool):
        """Refresh text only: never lift, deiconify, or focus a playing user."""
        if isinstance(labels, Mapping):
            parts = [f"{key}: {value}" for key, value in labels.items() if value]
            recognition = " · ".join(parts) or "No active gesture"
        else:
            recognition = str(labels) or "No active gesture"
        mode = "INVENTORY" if inventory_open else "GAMEPLAY"
        self.state_text.set(f"{mode} · {recognition}")

    def show(self):
        """Explicit user-open action may bring the guide forward."""
        self.deiconify()
        self.lift()
