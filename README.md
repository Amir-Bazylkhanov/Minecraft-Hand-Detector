# Minecraft Hand Detector — Minecraft Java Edition hand controls

Minecraft Hand Detector is a local Windows companion app for Minecraft **Java Edition**: a camera-based control companion, **not a server plugin and not a game mod** — nothing is installed into Minecraft or any server. It uses your connected Xiaomi camera to control the game through complete hand poses. Your **left hand moves the player**; your **right hand looks around, mines, uses items, and selects hotbar slots**. A compact, semi-transparent gesture guide shows numbered pose instructions and live recognition while you play.

Practice mode starts enabled. It recognizes gestures without sending input. Live controls require explicit arming and a focused Minecraft Java window. **F8 releases inputs and stops tracking.**

## Requirements and launch

- Windows 10/11 and Python **3.13** (developed on 3.13.6).
- Minecraft **Java Edition**, running under `javaw.exe` or `java.exe`.
- A connected camera exposed to Windows, such as your Xiaomi phone.
- Dependencies pinned in `requirements.txt`: `mediapipe==1.0.1`, `opencv-contrib-python==5.0.0.93`, and `cv2-enumerate-cameras==1.4.0`.

Run **run.bat**. It creates a local `.venv`, installs/checks dependencies, and starts the app. Do not install plain `opencv-python` alongside contrib: they share conflicting `cv2` files. The launcher repairs that conflict.

Manual launch:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

The MediaPipe `hand_landmarker.task` model downloads from Google's official model bucket the first time you select **Start tracking**, then stays cached in `models/`. That first download needs internet. Tracking and gesture processing run locally; camera frames are not uploaded.

## Set up before playing

1. Select your **Xiaomi camera**, then select **Start tracking**. Keep both hands visible and test poses in Practice mode.
2. Hold your wrists in comfortable neutral positions and select **Calibrate neutral hand positions**. Calibration captures each visible wrist's center and disarms controls; calibrate before arming. Moving away from these centers controls movement and looking.
3. Open **Gesture guide (keep beside game)**. Its compact rows show finger numbers and live recognized actions. It starts at 55% opacity; adjust the **Opacity** slider from 25% to fully opaque. Windowed or borderless Minecraft is recommended so the guide remains visible. **On top** is optional. Live updates do not bring the guide forward or steal focus.
4. Close Minecraft's inventory and use **Mark closed** (or **Mark open** if it is open) to synchronize the app's inventory belief.
5. Uncheck **Practice mode**, select **Arm**, and switch to Minecraft during the three-second countdown. Resume gameplay if the game is paused. Controls only send input while Minecraft Java has focus.

If you click the guide or another application after live control starts, focus loss disarms controls. Synchronize inventory and arm again before returning to gameplay.

## Gesture map

Finger numbers: **thumb 4 · index 8 · middle 12 · ring 16 · pinky 20**. These identify fingers on the preview and guide. Complete poses determine actions; fingertip-to-palm contacts are no longer the gameplay controls.

| Hand and complete pose | Action |
|---|---|
| **Left index only extended**, other fingers folded | Hand joystick: move wrist up/down/left/right from its calibrated center for **W/S/A/D**. Diagonals work; the center dead zone stops movement. |
| Same left pointing pose, wrist farther forward/up | **Sprint** while moving forward. Return to the inner zone to walk. |
| Left hand with **thumb extended** — thumb only, while pointing, or in a V | **Hold Space** immediately to keep jumping, independent of the movement center. Fold the thumb to release. |
| **Left V sign**: index and middle extended, ring and pinky folded | Use the same joystick while **sneaking**. |
| **Right index only extended**, other fingers folded | Move wrist around its center to **look**. Center stops turning. In inventory, the **cursor stays where it was when the menu opened**; pointing settles a neutral center on your wrist (hold still briefly), then moving the wrist from that center moves the cursor at a **constant speed** — pushing the wrist farther never speeds it up — set by the **Inventory cursor speed** slider (**default 80 px/sec**; lower it for precise slot selection), and the small center dead zone keeps it still. **Thumb out then in left-clicks** (not the V sign or middle finger). |
| **Right thumb extended then folded** — with a fist or while pointing (index may be extended for look + attack; middle/ring/pinky folded — not V, full open hand, or hotbar 🤙) | **Mine/attack**. One out-in cycle sends one timed left-click (one attack). A second cycle within **0.6 s** starts holding attack for continuous mining; each further cycle refreshes the hold. Mining ends **0.8 s** after the last cycle, or immediately on opening your hand, tracking loss, or entering a menu. After the hold ends, the next thumb out-in cycle sends a single attack click again. |
| **Right V sign**, briefly held | **Place/use** with a right-click in the world. Leave the pose before repeating. |
| **Both open palms shown together** | Toggle **inventory** instantly on the first recognized frame. Change pose to rearm. A single open palm or a fist does not open/close inventory. |
| **Right thumb and pinky extended**, middle three folded | Enter hotbar selection. **Fold thumb → previous/left slot; fold pinky → next/right slot.** Restore both fingers before another step. Hotbar mode takes priority: a folded thumb steps slots, never attacks. |

An open hand is neutral except when both open palms deliberately form the inventory chord. Pose changes must settle before applicable actions trigger, reducing accidental clicks during transitions. Inventory mode suppresses world movement, mining, and hotbar selection while providing cursor and click controls. The inventory cursor is relative, not absolute: it never teleports to your fingertip — it stays where it was on entry and follows wrist motion from the settled center, and the preview shows that center, its dead zone, and your wrist marker.

The gameplay map is currently defined in code, rather than editable through the old palm-contact configuration JSON. Legacy binding/editor modules remain for compatibility tests; their saved assignments do not customize this new pose map.

## Camera and safety controls

- **Camera / Refresh:** lists full device names from DirectShow and Media Foundation without opening cameras during discovery. A Xiaomi-family device is preferred on first launch. Selection is saved in `handcraft_camera.json` and resolved by stable identity at each start. If the selected phone disconnects, Minecraft Hand Detector reports an error; it never silently switches to the laptop camera or index zero. Changing camera or refreshing stops tracking and disarms first.
- **Practice mode:** default on; sends no real input. Switching modes disarms, requiring fresh arming for live input.
- **Arm / Disarm:** live arming requires known inventory state and provides a three-second countdown to switch to Minecraft.
- **Mark open / Mark closed:** inventory state is a belief, not read from Minecraft. Synchronize it if you open or close inventory with the keyboard.
- **Sensitivity:** legacy wrist-detection setting; thumb-cycle attacks use fixed extend/fold thresholds.
- **Inventory cursor speed:** cursor speed in **px/sec** while pointing inside an open inventory, from 20 to 240 in steps of 10 (**default 80 px/sec**). Lower values are slower, for precise slot selection between nearby slots. The cursor keeps its constant-speed behavior at every setting: the wrist only picks the direction, never the magnitude. Independent of the legacy Sensitivity slider.
- **Thumb detector feedback:** the camera preview shows a live thumb readout — **OUT / FOLDED / BETWEEN** — from the thumb-tip-to-index-knuckle gap normalized by palm size. Modern controls share one hysteresis pair: a gap above **0.65** registers OUT (extended), below **0.55** registers FOLDED, and inside the 0.55–0.65 band the detector retains its previous state, so mid-band jitter neither starts nor stops an attack.
- **Stop tracking / F8 / closing the app:** releases inputs and disarms. F8 works while Minecraft has focus and while initial model preparation is running.

Real input uses native Windows `SendInput`. The foreground title must contain “Minecraft” and its executable must be `javaw.exe` or `java.exe`; focus is checked immediately before input dispatch. Focus loss releases inputs, disarms, and makes inventory state unknown. Tracking loss, stale frames older than 300 ms, and errors also release controls. Input releases apply to controls the app pressed itself.

## Development and verification

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -t . -v
```

Main components:

- `posemap.py`: complete-pose recognition, movement/look centers, jump, sneak, sprint, use, inventory chord, inventory wrist cursor, and hotbar stepping.
- `posedispatch.py`: routes pose results through existing safety gates.
- `guide.py`: compact semi-transparent passive Tk gesture reference.
- `controller.py`, `adapters.py`, `win32input.py`: input lifecycle, timed key pulses, pointer controls, and OS injection.
- `tracker.py`, `cameras.py`: local landmark tracking and explicit named-camera selection.
- `focus.py`, `app.py`: foreground checks, emergency stop, and the single-root Tk UI.
- `geometry.py`, `gestures.py`: hand geometry and mining state logic.
- `bindings.py`, `handmap.py`: retained legacy contact bindings/editor and their tests.

Automated checks use synthetic landmarks and mock input adapters. A Tk guide smoke check confirms construction, live labels, hidden-window updates, and reopening. **Actual camera-to-Minecraft response for the new gesture map remains unverified**; try Practice mode before live play.

Thumb-cycle detection needs the thumb clearly visible against the curled fingers in the two-dimensional camera image. Keep the whole hand in frame and use clear poses; occlusion and camera angle affect recognition. Keyboard/mouse controls remain available for actions such as chat, pause, and dropping items.
