from ultralytics import YOLOWorld
import cv2
from threading import Thread
import torch
import numpy as np
from collections import deque
import threading
import time

from flask import Flask, render_template_string
from flask_socketio import SocketIO

# ===========================
# Flask + SocketIO Setup
# ===========================
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

def update_command(cmd):
    socketio.emit("command", cmd)

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <title>DivyaDrishti Navigation</title>
  <script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
  <style>
    body {
      background: #0d0d0d;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 100vh;
      margin: 0;
      font-family: monospace;
    }
    #cmd {
      font-size: 5rem;
      font-weight: bold;
      color: #00ff99;
      letter-spacing: 0.1em;
      text-align: center;
    }
    #zone-grid {
      display: grid;
      grid-template-columns: repeat(3, 120px);
      grid-template-rows: repeat(4, 60px);
      gap: 3px;
      margin-top: 1.5rem;
    }
    .zone-cell {
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.65rem;
      color: #888;
      border: 1px solid #222;
      border-radius: 3px;
      background: #111;
      transition: background 0.2s, color 0.2s;
    }
    .zone-cell.clear   { background: #0a2e0a; color: #00ff99; }
    .zone-cell.slight  { background: #2e2200; color: #ffaa00; }
    .zone-cell.danger  { background: #2e0000; color: #ff4444; }
    .zone-cell.duck    { background: #1a001a; color: #cc44ff; }
    #start-btn {
      font-size: 1.2rem;
      padding: 0.8rem 2rem;
      background: #00ff99;
      color: #0d0d0d;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      font-family: monospace;
      font-weight: bold;
      margin-top: 1.5rem;
    }
    #start-btn:disabled { background: #1a1a1a; color: #444; cursor: default; }
    #dot {
      width: 12px; height: 12px;
      border-radius: 50%;
      background: #ff3333;
      margin-top: 1rem;
    }
    #dot.connected { background: #00ff99; }
    #status { color: #555; font-size: 0.85rem; margin-top: 0.4rem; }
  </style>
</head>
<body>
  <div id="cmd">WAITING...</div>

  <div id="zone-grid">
    <div class="zone-cell" id="z00">TL</div>
    <div class="zone-cell" id="z01">TC</div>
    <div class="zone-cell" id="z02">TR</div>
    <div class="zone-cell" id="z10">UML</div>
    <div class="zone-cell" id="z11">UMC</div>
    <div class="zone-cell" id="z12">UMR</div>
    <div class="zone-cell" id="z20">LML</div>
    <div class="zone-cell" id="z21">LMC</div>
    <div class="zone-cell" id="z22">LMR</div>
    <div class="zone-cell" id="z30">BL</div>
    <div class="zone-cell" id="z31">BC</div>
    <div class="zone-cell" id="z32">BR</div>
  </div>

  <button id="start-btn">TAP TO ENABLE AUDIO</button>
  <div id="dot"></div>
  <div id="status">connecting...</div>

  <script>
    const socket   = io();
    const cmdEl    = document.getElementById('cmd');
    const dotEl    = document.getElementById('dot');
    const statEl   = document.getElementById('status');
    const startBtn = document.getElementById('start-btn');
    let audioUnlocked = false;

    const ZONE_IDS = [
      ['z00','z01','z02'],
      ['z10','z11','z12'],
      ['z20','z21','z22'],
      ['z30','z31','z32']
    ];

    startBtn.addEventListener('click', () => {
      speechSynthesis.speak(new SpeechSynthesisUtterance(''));
      audioUnlocked = true;
      startBtn.textContent = 'AUDIO ENABLED';
      startBtn.disabled = true;
    });

    socket.on('connect', () => {
      dotEl.classList.add('connected');
      statEl.textContent = 'connected';
    });
    socket.on('disconnect', () => {
      dotEl.classList.remove('connected');
      statEl.textContent = 'disconnected';
    });

    socket.on('command', (data) => {
      cmdEl.textContent = data.cmd.toUpperCase();
      if (audioUnlocked) {
        speechSynthesis.cancel();
        speechSynthesis.speak(new SpeechSynthesisUtterance(data.cmd));
      }
    });

    socket.on('zones', (grid) => {
      // grid is 4x3 array of depth values 0-1
      for (let r = 0; r < 4; r++) {
        for (let c = 0; c < 3; c++) {
          const el  = document.getElementById(ZONE_IDS[r][c]);
          const val = grid[r][c];
          el.className = 'zone-cell';
          if (val >= 0.85)      el.classList.add('duck');
          else if (val >= 0.70) el.classList.add('danger');
          else if (val >= 0.40) el.classList.add('slight');
          else                  el.classList.add('clear');
          el.title = val.toFixed(2);
        }
      }
    });
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


# ===========================
# Frame Dimensions
# ===========================
FRAME_W = 720
FRAME_H = 480

# Zone boundaries (row, col pixel ranges)
# 4 rows x 3 cols = 12 zones
ROW_BOUNDS = [(0, 120), (120, 240), (240, 360), (360, 480)]
COL_BOUNDS = [(0, 240), (240, 480), (480, 720)]

ZONE_NAMES = [
    ["TOP-LEFT",       "TOP-CENTER",       "TOP-RIGHT"      ],
    ["UPPER-MID-LEFT", "UPPER-MID-CENTER", "UPPER-MID-RIGHT"],
    ["LOWER-MID-LEFT", "LOWER-MID-CENTER", "LOWER-MID-RIGHT"],
    ["BOTTOM-LEFT",    "BOTTOM-CENTER",    "BOTTOM-RIGHT"   ],
]

# Depth thresholds
THR_SLIGHT_LO = 0.40
THR_SLIGHT_HI = 0.60
THR_TURN_LO   = 0.70
THR_TURN_HI   = 0.85
THR_CRITICAL  = 0.85

# Expected depth for top row (far ground) — anomaly = actual - expected
TOP_ROW_EXPECTED = 0.25
DUCK_ANOMALY_THR = 0.45

COMMAND_PHRASES = {
    "go":             "Go forward",
    "slightly_left":  "Slightly left",
    "slightly_right": "Slightly right",
    "left":           "Turn left",
    "right":          "Turn right",
    "duck":           "Duck down",
    "duck_left":      "Duck and go left",
    "duck_right":     "Duck and go right",
    "stop":           "Stop, obstacle ahead",
    "uturn":          "Make a U-turn",
}


# ===========================
# Webcam Stream Thread
# ===========================
class WebcamStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG) if isinstance(src, str) \
                   else cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.grabbed, self.frame = self.cap.read()
        self.stopped = False
        Thread(target=self._update, daemon=True).start()

    def _update(self):
        while not self.stopped:
            self.grabbed, self.frame = self.cap.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()


# ===========================
# Load Models
# ===========================
cv2.setNumThreads(6)
cv2.ocl.setUseOpenCL(True)

print("[INFO] Loading YOLO-World...")
model = YOLOWorld("yolov8s-world.pt")
with open("yolo_worlds_objects.txt", "r") as f:
    VOCAB = eval(f.read())
model.set_classes(VOCAB)
print("[INFO] YOLO-World loaded")

device = torch.device("cpu")
print("[INFO] Loading MiDaS...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(device)
midas.eval()
midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
print("[INFO] MiDaS loaded")


# ===========================
# MiDaS Background Worker
# ===========================
depth_map     = None
depth_lock    = threading.Lock()
depth_running = False

def depth_worker(frame_rgb):
    global depth_map, depth_running
    try:
        inp = midas_transform(frame_rgb).to(device)
        if inp.ndim == 3:
            inp = inp.unsqueeze(0)
        with torch.no_grad():
            dm = midas(inp).squeeze().cpu().numpy()
        dm = (dm - dm.min()) / (dm.max() - dm.min() + 1e-6)
        # Resize to exactly 480x720 (H x W)
        dm = cv2.resize(dm, (FRAME_W, FRAME_H), interpolation=cv2.INTER_LINEAR)
        with depth_lock:
            depth_map = dm
    finally:
        depth_running = False


# ===========================
# Zone Depth Calculator
# ===========================
def compute_zone_grid(depth):
    """
    Returns 4x3 numpy array of mean depth per zone.
    Rows: top, upper-mid, lower-mid, bottom
    Cols: left, center, right
    """
    grid = np.zeros((4, 3), dtype=np.float32)
    for r, (r0, r1) in enumerate(ROW_BOUNDS):
        for c, (c0, c1) in enumerate(COL_BOUNDS):
            grid[r, c] = float(np.mean(depth[r0:r1, c0:c1]))
    return grid


# ===========================
# Zone-Based Navigation Engine
# ===========================
def decide_command(grid):
    """
    12-zone navigation decision engine.

    Zone layout (row x col):
      Row 0: TOP          — overhead duck detection
      Row 1: UPPER-MID    — approach warnings
      Row 2: LOWER-MID    — slight turn triggers
      Row 3: BOTTOM       — immediate turn / stop triggers

    Depth scale: 0.0 = far/clear, 1.0 = near/obstacle

    Decision priority (highest first):
      1. DUCK  — top-center anomaly or critical depth anywhere top row
      2. STOP  — bottom-center >= 0.85 (collision imminent)
      3. TURN  — bottom L/R >= 0.70 (immediate obstacle)
      4. SLIGHT— lower-mid L/R in 0.40-0.60 (approaching)
      5. GO    — all zones clear
    """

    # ── Row extracts ──────────────────────────────────────────
    top_l,  top_c,  top_r  = grid[0]   # Row 0: overhead
    uml,    umc,    umr    = grid[1]   # Row 1: upper-mid
    lml,    lmc,    lmr    = grid[2]   # Row 2: lower-mid
    bot_l,  bot_c,  bot_r  = grid[3]   # Row 3: bottom

    # ── Priority 1: DUCK detection ────────────────────────────
    # Top-center is the primary duck zone
    # Anomaly = how much higher than expected far-ground depth
    top_c_anomaly = top_c - TOP_ROW_EXPECTED

    if top_c_anomaly > DUCK_ANOMALY_THR or top_c >= THR_CRITICAL:
        # Determine if user can dodge sideways
        if top_l < top_r:
            return "duck_left"
        elif top_r < top_l:
            return "duck_right"
        return "duck"

    # Wide overhead obstacle (two or more top zones anomalous)
    top_anomalies = sum(1 for v in [top_l, top_c, top_r]
                        if (v - TOP_ROW_EXPECTED) > 0.30)
    if top_anomalies >= 2:
        return "duck"

    # ── Priority 2: STOP — bottom center critical ─────────────
    if bot_c >= THR_CRITICAL:
        # Last resort: try to dodge
        if bot_l < bot_r:
            return "right"
        elif bot_r < bot_l:
            return "left"
        return "stop"

    # ── Priority 3: TURN — bottom row immediate obstacles ─────
    # Uses the exact depth-range table from the spec:
    # L>=0.70 and R<0.70  → Turn right (obstacle left)
    # R>=0.70 and L<0.70  → Turn left  (obstacle right)
    # Both >=0.70         → choose clearer side
    bot_l_danger = bot_l >= THR_TURN_LO
    bot_r_danger = bot_r >= THR_TURN_LO

    if bot_l_danger and bot_r_danger:
        return "right" if bot_l >= bot_r else "left"
    if bot_l_danger:
        return "right"
    if bot_r_danger:
        return "left"

    # ── Priority 4: SLIGHT — lower-mid approaching objects ────
    # Also check upper-mid for early warning merge
    # Lower-mid left/right in 0.40-0.60 → slight adjustment
    lml_slight = THR_SLIGHT_LO <= lml <= THR_SLIGHT_HI
    lmr_slight = THR_SLIGHT_LO <= lmr <= THR_SLIGHT_HI

    # Upper-mid amplifies the signal if both rows agree
    uml_warn = uml >= THR_SLIGHT_LO
    umr_warn = umr >= THR_SLIGHT_LO

    if lml_slight and lmr_slight:
        # Both sides approaching — go toward clearer side
        return "slightly_right" if lml >= lmr else "slightly_left"
    if lml_slight or (uml_warn and lml >= THR_SLIGHT_LO):
        return "slightly_right"
    if lmr_slight or (umr_warn and lmr >= THR_SLIGHT_LO):
        return "slightly_left"

    # ── Priority 5: Center path check ─────────────────────────
    # Lower-mid center blocked but not critical
    if lmc >= THR_TURN_LO:
        return "right" if lml < lmr else "left"
    if lmc >= THR_SLIGHT_LO:
        return "slightly_right" if lml < lmr else "slightly_left"

    # ── All clear ─────────────────────────────────────────────
    return "go"


# ===========================
# Stability Filter
# ===========================
class CommandFilter:
    """
    Emits a command only when it has been stable for N consecutive
    frames AND the cooldown since last emission has elapsed.
    Prevents single-frame noise from triggering voice output.
    """
    def __init__(self, stable_frames=3, cooldown_sec=3.0):
        self.stable_frames  = stable_frames
        self.cooldown_sec   = cooldown_sec
        self.history        = deque(maxlen=stable_frames)
        self.last_emitted   = None
        self.last_emit_time = 0.0

    def update(self, command):
        """
        Returns the command to emit, or None if not yet stable / in cooldown.
        """
        self.history.append(command)

        # Not enough history yet
        if len(self.history) < self.stable_frames:
            return None

        # All recent frames agree?
        if len(set(self.history)) != 1:
            return None

        stable_cmd = command
        now = time.time()

        # Same command still in cooldown
        if stable_cmd == self.last_emitted:
            if (now - self.last_emit_time) < self.cooldown_sec:
                return None

        # Emit
        self.last_emitted   = stable_cmd
        self.last_emit_time = now
        return stable_cmd


# ===========================
# Main Vision Loop
# ===========================
def vision_loop(stream):
    global depth_running

    cmd_filter   = CommandFilter(stable_frames=3, cooldown_sec=3.0)
    frame_count  = 0
    last_yolo    = None
    YOLO_EVERY_N = 2
    DEPTH_EVERY_N = 4

    while True:
        frame = stream.read()
        if frame is None:
            time.sleep(0.01)
            continue

        frame_count += 1

        # Resize to working resolution
        frame_720 = cv2.resize(frame, (FRAME_W, FRAME_H))

        # ── YOLO inference ────────────────────────────────────
        if frame_count % YOLO_EVERY_N == 0:
            last_yolo = model.predict(frame_720, conf=0.30, verbose=False)[0]

        # ── MiDaS depth (background thread) ──────────────────
        if frame_count % DEPTH_EVERY_N == 0 and not depth_running:
            depth_running = True
            frame_rgb = cv2.cvtColor(
                cv2.resize(frame, (384, 256)), cv2.COLOR_BGR2RGB
            )
            Thread(target=depth_worker, args=(frame_rgb,), daemon=True).start()

        # ── Navigation decision ───────────────────────────────
        with depth_lock:
            current_depth = depth_map

        if current_depth is None:
            # Depth not ready yet — show feed and wait
            cv2.imshow("DivyaDrishti", frame_720)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        grid       = compute_zone_grid(current_depth)
        raw_cmd    = decide_command(grid)
        emit_cmd   = cmd_filter.update(raw_cmd)

        if emit_cmd is not None:
            phrase = COMMAND_PHRASES.get(emit_cmd, emit_cmd)
            print(f"[NAV] {emit_cmd.upper():15s} → {phrase}")
            update_command({"cmd": phrase})
            # Send zone grid to dashboard for visualization
            socketio.emit("zones", grid.tolist())

        # ── OpenCV display ────────────────────────────────────
        display = last_yolo.plot() if last_yolo is not None else frame_720.copy()

        # Draw 12-zone grid overlay on display frame
        for r, (r0, r1) in enumerate(ROW_BOUNDS):
            for c, (c0, c1) in enumerate(COL_BOUNDS):
                val = grid[r, c]
                if val >= THR_CRITICAL:
                    color = (0, 0, 255)      # Red
                elif val >= THR_TURN_LO:
                    color = (0, 100, 255)    # Orange
                elif val >= THR_SLIGHT_LO:
                    color = (0, 220, 255)    # Yellow
                else:
                    color = (0, 180, 0)      # Green
                cv2.rectangle(display, (c0, r0), (c1, r1), color, 1)
                cv2.putText(display, f"{val:.2f}", (c0+4, r0+16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

        # Command overlay
        phrase = COMMAND_PHRASES.get(raw_cmd, raw_cmd)
        cv2.putText(display, f"CMD: {phrase}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)

        cv2.imshow("DivyaDrishti", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.stop()
    cv2.destroyAllWindows()


# ===========================
# Entry Point
# ===========================
if __name__ == "__main__":
    src = input("Enter IP webcam URL or 0 for laptop cam: ")
    if src == "0":
        src = 0

    stream = WebcamStream(src)

    vision_thread = Thread(target=vision_loop, args=(stream,), daemon=True)
    vision_thread.start()

    print("[INFO] Dashboard → http://localhost:5000")
    socketio.run(app, host="0.0.0.0", port=5000, use_reloader=False)
