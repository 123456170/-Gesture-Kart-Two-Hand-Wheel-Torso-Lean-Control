import json
import math
import os
import random
import tempfile
import threading
import time
from collections import deque

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
    WEBRTC_OK = True
except Exception:
    VideoProcessorBase = object
    webrtc_streamer = None
    WEBRTC_OK = False

try:
    import mediapipe as mp
    MP_OK = True
except Exception:
    mp = None
    MP_OK = False

st.set_page_config(
    page_title="Air Flick Gesture Runner",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEMO_MODE = "Auto Demo (instant)"
REAL_MODE = "Real Webcam"

GESTURE_TO_KEY = {
    "Up": "ArrowUp",
    "Down": "ArrowDown",
    "Left": "ArrowLeft",
    "Right": "ArrowRight",
}

DIRECTION_VECTORS = {
    "Up": np.array([0.0, -1.0], dtype=np.float32),
    "Down": np.array([0.0, 1.0], dtype=np.float32),
    "Left": np.array([-1.0, 0.0], dtype=np.float32),
    "Right": np.array([1.0, 0.0], dtype=np.float32),
}


@st.cache_resource(show_spinner=False)
def get_store():
    return {
        "events": [],
        "stats": {"accepted": 0, "rejected": 0},
        "lock": threading.RLock(),
        "next_id": 0,
        "config": {
            "threshold": 1.35,
            "cooldown": 0.35,
            "buffer_len": 6,
            "calibrating": True,
            "calibrated": False,
            "record": False,
        },
        "calib": {
            "samples": [],
            "neutral": None,
            "scale": None,
        },
        "recorder": {
            "writer": None,
            "path": None,
            "ready": False,
        },
    }


STORE = get_store()


def add_event(gesture, latency_ms, confidence, source, accepted=True, note=""):
    with STORE["lock"]:
        STORE["next_id"] += 1
        event = {
            "id": STORE["next_id"],
            "epoch": time.time(),
            "time": time.strftime("%H:%M:%S"),
            "gesture": gesture,
            "latency_ms": int(latency_ms),
            "confidence": round(float(confidence), 2),
            "source": source,
            "accepted": bool(accepted),
            "note": note,
        }
        STORE["events"].append(event)
        if len(STORE["events"]) > 500:
            del STORE["events"][:-500]

        if accepted:
            STORE["stats"]["accepted"] += 1
        else:
            STORE["stats"]["rejected"] += 1
        return event


def get_events(limit=None):
    with STORE["lock"]:
        events = list(STORE["events"])
    if limit is not None:
        events = events[-limit:]
    return events


def finalize_recording():
    rec = STORE["recorder"]
    if rec.get("writer") is not None:
        try:
            rec["writer"].release()
        except Exception:
            pass
        rec["writer"] = None
        rec["ready"] = True


def ensure_recorder(frame, fps=20.0):
    cfg = STORE["config"]
    rec = STORE["recorder"]

    if cfg.get("record"):
        if rec.get("writer") is None:
            h, w = frame.shape[:2]
            path = os.path.join(
                tempfile.gettempdir(),
                f"air_flick_session_{time.strftime('%Y%m%d_%H%M%S')}.mp4",
            )
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(path, fourcc, float(fps), (w, h))
            if writer.isOpened():
                rec["writer"] = writer
                rec["path"] = path
                rec["ready"] = False
            else:
                try:
                    writer.release()
                except Exception:
                    pass
                cfg["record"] = False
                return

        if rec.get("writer") is not None:
            rec["writer"].write(frame)
    else:
        if rec.get("writer") is not None:
            finalize_recording()


GAME_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<style>
  html, body {
    margin: 0;
    padding: 0;
    background: #0f1420;
    color: #eaf2ff;
    font-family: Arial, Helvetica, sans-serif;
    overflow: hidden;
  }
  .wrap {
    width: 100%;
    height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-start;
    padding: 10px;
    box-sizing: border-box;
  }
  h3 {
    margin: 4px 0 8px 0;
    font-size: 16px;
    color: #9fe8ff;
  }
  canvas {
    border-radius: 12px;
    box-shadow: 0 0 24px rgba(0, 255, 200, 0.12);
    background: #151b29;
  }
  .hint {
    margin-top: 8px;
    font-size: 12px;
    color: #a7b4d0;
    text-align: center;
    line-height: 1.35;
  }
</style>
</head>
<body>
  <div class="wrap">
    <h3>Runner Game Viewport</h3>
    <canvas id="game" width="360" height="430"></canvas>
    <div class="hint">
      Built-in same-origin game receives arrow-key events dispatched by the gesture pipeline.<br/>
      Up = jump, Down = slide, Left/Right = change lane.
    </div>
  </div>

<script>
  const canvas = document.getElementById("game");
  const ctx = canvas.getContext("2d");

  const state = {
    lane: 1,
    y: 0,
    vy: 0,
    slideUntil: 0,
    last: "None",
    count: 0,
  };

  const seen = new Set();

  function handleKey(key) {
    if (key === "ArrowLeft") {
      state.lane = Math.max(0, state.lane - 1);
      state.last = "Left";
      state.count++;
    } else if (key === "ArrowRight") {
      state.lane = Math.min(2, state.lane + 1);
      state.last = "Right";
      state.count++;
    } else if (key === "ArrowUp") {
      if (state.y === 0) {
        state.vy = -11.0;
      }
      state.last = "Up";
      state.count++;
    } else if (key === "ArrowDown") {
      state.slideUntil = Date.now() + 450;
      state.last = "Down";
      state.count++;
    }
  }

  function receive(payload) {
    if (!payload || !payload.key) return;
    const id = payload.id || JSON.stringify(payload);
    if (seen.has(id)) return;
    seen.add(id);
    if (seen.size > 300) {
      seen.delete(seen.keys().next().value);
    }
    handleKey(payload.key);
  }

  window.addEventListener("keydown", function(e) {
    if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(e.key)) {
      e.preventDefault();
      handleKey(e.key);
    }
  });

  try {
    const bc = new BroadcastChannel("air_flick_gestures");
    bc.onmessage = function(e) {
      receive(e.data);
    };
  } catch (err) {}

  setInterval(function() {
    try {
      while (window.__airFlickQueue && window.__airFlickQueue.length) {
        receive(window.__airFlickQueue.shift());
      }
    } catch (err) {}

    try {
      if (window.parent && window.parent !== window) {
        while (window.parent.__airFlickQueue && window.parent.__airFlickQueue.length) {
          receive(window.parent.__airFlickQueue.shift());
        }
      }
    } catch (err) {}

    try {
      const raw = localStorage.getItem("air_flick_last");
      if (raw) {
        const payload = JSON.parse(raw);
        receive(payload);
      }
    } catch (err) {}
  }, 40);

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = "#151b29";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    for (let i = 0; i <= 3; i++) {
      const x = 30 + i * 100;
      ctx.strokeStyle = "rgba(120, 160, 255, 0.18)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
    }

    const markerOffset = (Date.now() / 18) % 42;
    ctx.strokeStyle = "rgba(255,255,255,0.16)";
    ctx.lineWidth = 3;
    for (let y = -42 + markerOffset; y < canvas.height; y += 42) {
      for (let lane = 0; lane < 3; lane++) {
        const x = 80 + lane * 100;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x, y + 18);
        ctx.stroke();
      }
    }

    state.vy += 0.58;
    state.y += state.vy;
    if (state.y > 0) {
      state.y = 0;
      state.vy = 0;
    }

    const groundY = 315;
    const sliding = Date.now() < state.slideUntil;
    const runnerX = 63 + state.lane * 100;
    const runnerY = groundY + state.y;
    const runnerW = 34;
    const runnerH = sliding ? 24 : 48;

    ctx.fillStyle = "#31d98c";
    ctx.beginPath();
    ctx.roundRect(runnerX, runnerY - runnerH, runnerW, runnerH, 8);
    ctx.fill();

    ctx.fillStyle = "#0f1420";
    ctx.beginPath();
    ctx.arc(runnerX + 17, runnerY - runnerH + 10, 6, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "#eaf2ff";
    ctx.font = "15px Arial";
    ctx.fillText("Last gesture: " + state.last, 16, 28);
    ctx.fillText("Events received: " + state.count, 16, 52);

    requestAnimationFrame(draw);
  }

  draw();
</script>
</body>
</html>
"""


def dispatch_js(gesture: str) -> str:
    key = GESTURE_TO_KEY.get(gesture)
    if not key:
        return "<div></div>"

    payload = {
        "key": key,
        "gesture": gesture,
        "id": str(time.time_ns()),
    }
    payload_json = json.dumps(payload)

    return """
<script>
(function() {
  const payload = __PAYLOAD__;

  try {
    window.__airFlickQueue = window.__airFlickQueue || [];
    window.__airFlickQueue.push(payload);
  } catch (e) {}

  try {
    if (window.parent && window.parent !== window) {
      window.parent.__airFlickQueue = window.parent.__airFlickQueue || [];
      window.parent.__airFlickQueue.push(payload);
    }
  } catch (e) {}

  try {
    localStorage.setItem("air_flick_last", JSON.stringify(payload));
  } catch (e) {}

  try {
    const bc = new BroadcastChannel("air_flick_gestures");
    bc.postMessage(payload);
    bc.close();
  } catch (e) {}

  try {
    const evt = new KeyboardEvent("keydown", {
      key: payload.key,
      code: payload.key,
      bubbles: true,
      cancelable: true
    });
    window.dispatchEvent(evt);

    setTimeout(() => {
      window.dispatchEvent(new KeyboardEvent("keyup", {
        key: payload.key,
        code: payload.key,
        bubbles: true
      }));
    }, 70);
  } catch (e) {}
})();
</script>
""".replace("__PAYLOAD__", payload_json)


class HandFlickProcessor(VideoProcessorBase):
    def __init__(self):
        super().__init__()
        self.hands = None
        if MP_OK:
            self.hands = mp.solutions.hands.Hands(
                max_num_hands=1,
                model_complexity=0,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            )

        self.buffer = deque(maxlen=int(STORE["config"].get("buffer_len", 6)))
        self.last_gesture_mono = 0.0
        self.last_noise_mono = 0.0
        self.last_badge = ""
        self.last_badge_mono = 0.0
        self.last_latency = 0

    def recv(self, frame):
        start = time.perf_counter()

        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        if img.shape[1] > 720:
            scale = 720.0 / float(img.shape[1])
            img = cv2.resize(img, (720, int(img.shape[0] * scale)))

        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb) if self.hands is not None else None

        cfg = STORE["config"]
        calib = STORE["calib"]
        now_mono = time.monotonic()

        hand_present = False
        point = None
        mp_score = 0.8

        if results and results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            hand_present = True

            try:
                mp_score = float(results.multi_handedness[0].classification[0].score)
            except Exception:
                mp_score = 0.8

            if MP_OK:
                try:
                    mp.solutions.drawing_utils.draw_landmarks(
                        img,
                        hand,
                        mp.solutions.hands.HAND_CONNECTIONS,
                    )
                except Exception:
                    pass

            lm = hand.landmark
            tip = lm[8]
            point = (int(tip.x * w), int(tip.y * h))

            wrist = lm[0]
            middle = lm[9]
            hand_px = math.hypot(
                (middle.x - wrist.x) * w,
                (middle.y - wrist.y) * h,
            )
            hand_frac = max(hand_px / float(min(h, w)), 0.02)

            if cfg.get("calibrating") and not cfg.get("calibrated"):
                calib["samples"].append((float(tip.x), float(tip.y), float(hand_frac)))
                if len(calib["samples"]) > 45:
                    calib["samples"] = calib["samples"][-45:]

                if len(calib["samples"]) >= 20:
                    arr = np.array(calib["samples"], dtype=np.float32)
                    std_xy = float(np.mean(np.std(arr[:, :2], axis=0)))
                    std_scale = float(np.std(arr[:, 2]))

                    if len(calib["samples"]) >= 30 or (std_xy < 0.012 and std_scale < 0.012):
                        calib["neutral"] = (
                            float(np.mean(arr[:, 0])),
                            float(np.mean(arr[:, 1])),
                        )
                        calib["scale"] = float(np.mean(arr[:, 2]))
                        cfg["calibrated"] = True
                        cfg["calibrating"] = False
                        self.buffer.clear()

        if not hand_present:
            self.buffer.clear()
            cv2.putText(
                img,
                "Show your hand to the camera",
                (20, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            if cfg.get("calibrating") and not cfg.get("calibrated"):
                cv2.putText(
                    img,
                    "Calibration waiting for a hand...",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

            ensure_recorder(img, fps=28.0)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cv2.circle(img, point, 10, (0, 255, 255), -1)

        if cfg.get("calibrating") and not cfg.get("calibrated"):
            progress = min(len(calib.get("samples", [])) / 30.0, 1.0)
            cv2.rectangle(
                img,
                (w // 2 - 120, h // 2 - 120),
                (w // 2 + 120, h // 2 + 120),
                (0, 255, 255),
                2,
            )
            cv2.putText(
                img,
                f"Calibrating: hold steady {int(progress * 100)}%",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            ensure_recorder(img, fps=28.0)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if cfg.get("calibrated") and calib.get("neutral") is not None and calib.get("scale") is not None:
            scale = max(float(calib.get("scale", 0.05)), 0.02)
            nx = (float(tip.x) - float(calib["neutral"][0])) / scale
            ny = (float(tip.y) - float(calib["neutral"][1])) / scale

            self.buffer.append((now_mono, nx, ny))

            if len(self.buffer) >= 4:
                t0, x0, y0 = self.buffer[0]
                t1, x1, y1 = self.buffer[-1]
                dt = max(t1 - t0, 1e-4)

                dx = x1 - x0
                dy = y1 - y0
                dist = math.hypot(dx, dy)
                speed = dist / dt

                threshold = float(cfg.get("threshold", 1.35))
                cooldown = float(cfg.get("cooldown", 0.35))

                if now_mono - self.last_gesture_mono > cooldown:
                    direction = None

                    if speed >= threshold and dist >= 0.08 and dt <= 0.45:
                        if abs(dx) > abs(dy) * 1.25:
                            direction = "Right" if dx > 0 else "Left"
                        elif abs(dy) > abs(dx) * 1.25:
                            direction = "Down" if dy > 0 else "Up"

                    if direction:
                        self.last_gesture_mono = now_mono
                        self.last_badge = direction
                        self.last_badge_mono = now_mono

                        latency = int((time.perf_counter() - start) * 1000.0 + 35.0)
                        speed_conf = min(1.0, max(0.0, speed / max(threshold, 1e-6)))
                        confidence = min(
                            1.0,
                            max(0.55, 0.55 * mp_score + 0.45 * speed_conf),
                        )

                        add_event(
                            direction,
                            latency,
                            confidence,
                            "webcam",
                            True,
                            "live flick",
                        )
                        self.buffer.clear()

                    elif (
                        speed >= threshold * 0.55
                        and dist >= 0.05
                        and now_mono - self.last_noise_mono > 1.0
                    ):
                        self.last_noise_mono = now_mono
                        latency = int((time.perf_counter() - start) * 1000.0 + 35.0)
                        add_event(
                            "Noise",
                            latency,
                            max(0.2, mp_score * 0.7),
                            "webcam",
                            False,
                            "ambiguous / near miss",
                        )

        process_ms = (time.perf_counter() - start) * 1000.0
        self.last_latency = int(process_ms + 30.0)

        cv2.putText(
            img,
            f"Latency: {self.last_latency} ms",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )

        if cfg.get("calibrated"):
            cv2.putText(
                img,
                "Calibrated",
                (w - 130, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 120),
                2,
            )

        if self.last_badge and now_mono - self.last_badge_mono < 1.2:
            cv2.putText(
                img,
                self.last_badge,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 255, 0),
                3,
            )

        ensure_recorder(img, fps=28.0)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def reset_demo():
    st.session_state["demo_start"] = time.time()
    st.session_state["demo_timeline"] = build_demo_timeline()
    st.session_state["demo_next_idx"] = 0
    st.session_state["demo_manuals"] = []


def build_demo_timeline():
    rng = random.Random(20)
    timeline = []

    t = 4.0
    for i, d in enumerate(["Up", "Left", "Down", "Right", "Up"]):
        timeline.append(
            {
                "t": t + i * 4.0,
                "dir": d,
                "kind": "flick",
                "amp": 115.0,
                "dur": 0.34,
            }
        )

    t = 26.0
    dirs = ["Up", "Down", "Left", "Right"]
    for i in range(36):
        timeline.append(
            {
                "t": t + i * 1.25,
                "dir": rng.choice(dirs),
                "kind": "flick",
                "amp": 130.0,
                "dur": 0.22,
            }
        )

    t = 73.0
    for i in range(9):
        timeline.append(
            {
                "t": t + i * 2.7,
                "dir": rng.choice(dirs),
                "kind": "noise",
                "amp": 30.0,
                "dur": 0.5,
            }
        )

    return timeline


def demo_offset(now):
    offset = np.array([0.0, 0.0], dtype=np.float32)

    timeline = st.session_state.get("demo_timeline", [])
    for e in timeline:
        dt = now - float(e["t"])
        dur = float(e.get("dur", 0.3))
        if 0.0 <= dt <= dur:
            vec = DIRECTION_VECTORS.get(e.get("dir"), np.array([0.0, 0.0], dtype=np.float32))
            amp = float(e.get("amp", 30.0))
            progress = dt / max(dur, 1e-6)
            offset = offset + vec * (amp * math.sin(math.pi * progress))

    manuals = st.session_state.get("demo_manuals", [])
    active = []
    for m in manuals:
        dt = now - float(m["start"])
        dur = float(m.get("dur", 0.3))

        if dt < 0.0:
            active.append(m)
            continue

        if dt <= dur:
            vec = DIRECTION_VECTORS.get(m.get("dir"), np.array([0.0, 0.0], dtype=np.float32))
            amp = float(m.get("amp", 120.0))
            offset = offset + vec * (amp * math.sin(math.pi * (dt / max(dur, 1e-6))))
            active.append(m)

    st.session_state["demo_manuals"] = active

    offset[0] += 9.0 * math.sin(now * 1.7)
    offset[1] += 7.0 * math.cos(now * 2.4)

    return offset


def process_demo_timeline(now):
    timeline = st.session_state.get("demo_timeline", [])
    idx = st.session_state.get("demo_next_idx", 0)

    while idx < len(timeline) and float(timeline[idx]["t"]) <= now:
        e = timeline[idx]

        if e.get("kind") == "flick":
            add_event(
                e.get("dir", "Up"),
                random.randint(24, 48),
                round(random.uniform(0.88, 0.99), 2),
                "demo",
                True,
                "scripted flick",
            )
        else:
            add_event(
                "Noise",
                random.randint(31, 62),
                round(random.uniform(0.35, 0.62), 2),
                "demo",
                False,
                "near-miss wobble",
            )

        idx += 1

    st.session_state["demo_next_idx"] = idx


def render_demo_frame(now):
    h, w = 480, 640
    img = np.full((h, w, 3), (26, 29, 40), dtype=np.uint8)

    for x in range(0, w, 48):
        cv2.line(img, (x, 0), (x, h), (38, 42, 58), 1)
    for y in range(0, h, 48):
        cv2.line(img, (0, y), (w, y), (38, 42, 58), 1)

    center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
    offset = demo_offset(now)
    tip = center + offset
    tip_int = (int(tip[0]), int(tip[1]))

    wrist = tip + np.array([20, 120], dtype=np.float32)
    palm = tip + np.array([2, 70], dtype=np.float32)

    cv2.line(img, tuple(wrist.astype(int)), tuple(palm.astype(int)), (0, 200, 190), 4)
    cv2.line(img, tuple(palm.astype(int)), tip_int, (0, 220, 255), 4)

    for j in range(-2, 3):
        knuckle = palm + np.array([j * 14, 18], dtype=np.float32)
        finger_mid = palm + np.array([j * 8, 60], dtype=np.float32)
        cv2.line(img, tuple(knuckle.astype(int)), tuple(finger_mid.astype(int)), (0, 180, 170), 2)

    cv2.circle(img, tuple(wrist.astype(int)), 10, (0, 200, 190), -1)
    cv2.circle(img, tip_int, 12, (0, 255, 255), -1)

    if now < 3.5:
        cv2.rectangle(
            img,
            (w // 2 - 130, h // 2 - 130),
            (w // 2 + 130, h // 2 + 130),
            (0, 255, 255),
            2,
        )
        cv2.putText(
            img,
            "Calibration: hold hand steady in box",
            (w // 2 - 240, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )

    events = get_events(10)
    last_accepted = None
    for e in reversed(events):
        if e.get("accepted"):
            last_accepted = e
            break

    if last_accepted and time.time() - float(last_accepted.get("epoch", 0.0)) < 1.2:
        cv2.putText(
            img,
            str(last_accepted.get("gesture", "")),
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            4,
        )

    latency = int(last_accepted.get("latency_ms", 0)) if last_accepted else int(18 + 8 * abs(math.sin(now * 2.1)))
    cv2.putText(
        img,
        f"Latency: {latency} ms",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        img,
        "Demo mode: synthetic hand + real event pipeline",
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (170, 180, 200),
        1,
    )

    ensure_recorder(img, fps=12.0)
    return img


def simulate_gesture(direction, accepted=True, source_name="manual-button"):
    if accepted:
        latency = random.randint(22, 48)
        confidence = random.uniform(0.90, 0.99)
        note = "manual button"
        gesture = direction
    else:
        latency = random.randint(35, 70)
        confidence = random.uniform(0.35, 0.65)
        note = "manual near-miss"
        gesture = "Noise"

    add_event(gesture, latency, confidence, source_name, accepted, note)

    now_rel = time.time() - st.session_state.get("demo_start", time.time())
    manuals = st.session_state.setdefault("demo_manuals", [])

    if accepted:
        manuals.append(
            {
                "start": now_rel,
                "dir": direction,
                "dur": 0.28,
                "amp": 125.0,
            }
        )
    else:
        manuals.append(
            {
                "start": now_rel,
                "dir": random.choice(list(DIRECTION_VECTORS.keys())),
                "dur": 0.45,
                "amp": 26.0,
            }
        )


@st.fragment(run_every=0.08)
def demo_view():
    if "demo_start" not in st.session_state:
        reset_demo()

    now = time.time() - st.session_state["demo_start"]
    process_demo_timeline(now)
    img = render_demo_frame(now)
    st.image(img, channels="BGR", use_container_width=True)


@st.fragment(run_every=0.15)
def gesture_dispatcher():
    last_seen = st.session_state.get("last_seen_event_id", 0)
    events = get_events()

    if not events:
        return

    max_id = max(e["id"] for e in events)
    if max_id <= last_seen:
        return

    new_events = [e for e in events if e["id"] > last_seen]
    st.session_state["last_seen_event_id"] = max_id

    accepted = [
        e for e in new_events
        if e["accepted"] and e["gesture"] in GESTURE_TO_KEY
    ]

    if accepted:
        components.html(dispatch_js(accepted[-1]["gesture"]), height=0)


@st.fragment(run_every=0.5)
def status_panel():
    stats = STORE["stats"]
    total = stats["accepted"] + stats["rejected"]
    accuracy = int(100 * stats["accepted"] / total) if total else 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Successful flicks", stats["accepted"])
    c2.metric("Missed / noise", stats["rejected"])
    c3.metric("Accuracy", f"{accuracy}%")

    events = get_events(14)[::-1]
    if events:
        df = pd.DataFrame(events)
        df = df[["time", "gesture", "latency_ms", "confidence", "source", "accepted", "note"]]
        df["confidence"] = (df["confidence"] * 100).round(0).astype(int).astype(str) + "%"
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No gesture events yet.")


st.sidebar.title("Air Flick Controls")

source = st.sidebar.radio(
    "Input source",
    [DEMO_MODE, REAL_MODE],
    index=0,
    help="Auto Demo starts instantly with realistic simulated gestures. Real Webcam uses MediaPipe Hands on your live camera.",
)

if source == DEMO_MODE:
    if st.session_state.get("last_source") != DEMO_MODE:
        reset_demo()
    st.session_state["last_source"] = DEMO_MODE
else:
    st.session_state["last_source"] = REAL_MODE

st.session_state.setdefault("last_seen_event_id", 0)

threshold = st.sidebar.slider(
    "Flick speed threshold (lower = more sensitive)",
    0.60,
    3.00,
    float(STORE["config"].get("threshold", 1.35)),
    0.05,
)
STORE["config"]["threshold"] = float(threshold)

cooldown_ms = st.sidebar.slider(
    "Cooldown after gesture (ms)",
    250,
    600,
    int(float(STORE["config"].get("cooldown", 0.35)) * 1000),
    25,
)
STORE["config"]["cooldown"] = float(cooldown_ms) / 1000.0

record = st.sidebar.checkbox(
    "Record session to MP4",
    value=bool(STORE["config"].get("record", False)),
)
if record != bool(STORE["config"].get("record", False)):
    STORE["config"]["record"] = bool(record)
    if not record:
        finalize_recording()

if st.sidebar.button("Finalize recording"):
    finalize_recording()

rec = STORE["recorder"]
if rec.get("ready") and rec.get("path") and os.path.exists(rec.get("path", "")) and rec.get("writer") is None:
    try:
        with open(rec["path"], "rb") as f:
            video_bytes = f.read()
        st.sidebar.download_button(
            "Download MP4 demo video",
            data=video_bytes,
            file_name=os.path.basename(rec["path"]),
            mime="video/mp4",
        )
    except Exception:
        st.sidebar.warning("Could not load MP4 file for download.")
elif rec.get("writer") is not None:
    st.sidebar.caption("Recording in progress...")

events_all = get_events()
if events_all:
    csv_bytes = pd.DataFrame(events_all).to_csv(index=False).encode("utf-8")
    st.sidebar.download_button(
        "Download gesture CSV",
        data=csv_bytes,
        file_name="gesture_events.csv",
        mime="text/csv",
    )

if st.sidebar.button("Clear log & stats"):
    with STORE["lock"]:
        STORE["events"] = []
        STORE["stats"] = {"accepted": 0, "rejected": 0}
        STORE["next_id"] = 0
    st.session_state["last_seen_event_id"] = 0

if source == DEMO_MODE:
    if st.sidebar.button("Reset demo timeline"):
        reset_demo()

if source == REAL_MODE:
    st.sidebar.markdown("### Calibration")

    if st.sidebar.button("Start calibration"):
        STORE["config"]["calibrating"] = True
        STORE["config"]["calibrated"] = False
        STORE["calib"]["samples"] = []

    if st.sidebar.button("Skip calibration (defaults)"):
        STORE["calib"]["neutral"] = (0.5, 0.5)
        STORE["calib"]["scale"] = 0.08
        STORE["config"]["calibrated"] = True
        STORE["config"]["calibrating"] = False

    status = "Calibrated" if STORE["config"].get("calibrated") else (
        "Calibrating..." if STORE["config"].get("calibrating") else "Not calibrated"
    )
    st.sidebar.caption(f"Calibration status: {status}")

st.title("Air Flick Gesture Runner")
st.caption(
    "Live fingertip flick detection with MediaPipe landmark tracking, calibration, gesture classification, "
    "cooldown logic, latency display, event logging, and browser-side arrow-key dispatch."
)

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Camera / Demo View")

    if source == DEMO_MODE:
        demo_view()
        st.caption(
            "Auto Demo is running instantly with realistic simulated gestures. "
            "Switch to Real Webcam to use your actual camera."
        )
    else:
        if not WEBRTC_OK:
            st.error("streamlit-webrtc is not available. Install dependencies from requirements.txt.")
        elif not MP_OK:
            st.warning("MediaPipe is not available in this environment. Real webcam tracking is disabled.")

        if WEBRTC_OK:
            webrtc_streamer(
                key="real-webcam",
                video_processor_factory=HandFlickProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            st.caption(
                "Press START in the webcam widget and allow camera access. "
                "Calibration begins automatically when your hand is visible."
            )

    st.markdown("**Manual test buttons**")
    mb = st.columns(5)

    if mb[0].button("Up", use_container_width=True):
        simulate_gesture("Up", accepted=True)

    if mb[1].button("Down", use_container_width=True):
        simulate_gesture("Down", accepted=True)

    if mb[2].button("Left", use_container_width=True):
        simulate_gesture("Left", accepted=True)

    if mb[3].button("Right", use_container_width=True):
        simulate_gesture("Right", accepted=True)

    if mb[4].button("Near-miss", use_container_width=True):
        simulate_gesture("Up", accepted=False)

with right:
    st.subheader("Game viewport")
    components.html(GAME_HTML, height=500, scrolling=False)

    gesture_dispatcher()

    st.subheader("Gesture log & accuracy")
    status_panel()

    st.caption(
        "Browser security note: synthetic keyboard events can reliably drive same-origin embedded content. "
        "Cross-origin tabs/iframes usually cannot receive injected events without the target page cooperating."
    )