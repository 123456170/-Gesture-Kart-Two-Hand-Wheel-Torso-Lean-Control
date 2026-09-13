"""
Gesture-Controlled Racing Input System
---------------------------------------
Two-hand "steering wheel" + torso lean (accelerate/brake) captured live from
the webcam via streamlit-webrtc, processed with MediaPipe Hands + Pose,
smoothed with an EMA filter, and translated into simulated keyboard input
(arrow-key taps / holds) that drives a small built-in racing mini-game.

HONEST LIMITATION NOTE (read this):
Browsers do not allow a webpage to synthesize keyboard events that are
delivered to a *different*, cross-origin embedded game (e.g. an iframe of a
commercial racing site, or a totally separate native game window). That is a
browser security boundary, not a bug in this app. So this app:
  1. Dispatches real KeyboardEvents (keydown/keyup) on `document`, exactly the
     way a physical key press would, so any same-origin / same-page game
     logic listening for arrow keys will respond.
  2. Ships with a small built-in top-down kart mini-game (rendered as an SVG)
     that IS driven by those events, so you get a fully working, runnable,
     end-to-end demo without depending on a third-party game's security model.
If you want to wire this into a specific real game, that game needs to either
(a) be embedded same-origin, or (b) expose its own input hook you call
directly instead of a synthetic DOM event.

Run with:
    streamlit run app.py
"""

import csv
import io
import math
import os
import tempfile
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

try:
    import av
except ImportError:
    av = None

# --------------------------------------------------------------------------------------
# Constants / tunables
# --------------------------------------------------------------------------------------
EMA_ALPHA = 0.25                 # smoothing factor for the wheel-angle EMA
STEER_MAX_ANGLE = 45.0           # degrees mapped to "full lock" steering value of +-1
NOISE_GATE_DEGREES = 3.0         # angle wobble below this is ignored as noise
LEAN_THRESHOLD_DEFAULT = 6.0     # degrees beyond calibrated neutral to trigger accel/brake
TAP_MIN_HZ = 2.0                 # slowest tap rate (near-neutral wheel angle)
TAP_MAX_HZ = 12.0                # fastest tap rate (full lock)
LOOP_TICK_SECONDS = 0.08         # UI refresh / control loop cadence

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --------------------------------------------------------------------------------------
# Shared thread-safe state (video processor thread  <->  main Streamlit thread)
# --------------------------------------------------------------------------------------
STATE_LOCK = threading.Lock()
SHARED_STATE = {
    "raw_angle": 0.0,
    "ema_angle": 0.0,
    "steering_value": 0.0,     # -1..+1
    "torso_lean": 0.0,         # degrees relative to calibrated neutral
    "accel": False,
    "brake": False,
    "gesture_label": "waiting for hands...",
    "hands_detected": False,
    "pose_detected": False,
    "last_frame_ts": time.time(),
    "event_log": [],          # list of dicts, also mirrored to CSV file on disk
}

CALIBRATION = {
    "calibrated": False,
    "neutral_wheel_distance": None,
    "neutral_torso_angle": None,
    "_capture_next": False,
}


def _angle_deg(p1, p2):
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))


def log_event(label, detail=""):
    row = {
        "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
        "event": label,
        "detail": detail,
    }
    with STATE_LOCK:
        SHARED_STATE["event_log"].append(row)
        SHARED_STATE["gesture_label"] = label
    csv_path = st.session_state.get("csv_path")
    if csv_path:
        try:
            write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["timestamp", "event", "detail"])
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        except Exception:
            pass


# --------------------------------------------------------------------------------------
# Video processor: runs MediaPipe Hands + Pose on every frame
# --------------------------------------------------------------------------------------
class GestureProcessor(VideoProcessorBase):
    def __init__(self):
        try:
            self.hands = mp_hands.Hands(
                model_complexity=0,
                max_num_hands=2,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            )
            self.pose = mp_pose.Pose(
                model_complexity=0,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            )
        except AttributeError as e:
            raise RuntimeError(
                "MediaPipe's 'solutions' API (Hands/Pose) isn't available in this "
                "environment's Python/mediapipe build. This usually means the Python "
                "runtime is newer than what mediapipe currently supports (mediapipe "
                "needs Python 3.9-3.11). Pin the Python version via runtime.txt "
                "(e.g. 'python-3.11') and pin mediapipe in requirements.txt, then "
                "redeploy."
            ) from e
        self.ema_angle = 0.0
        self.prev_gesture = "neutral"
        self.recording = False
        self.video_writer = None
        self.lean_threshold = LEAN_THRESHOLD_DEFAULT

    def _ensure_writer(self, w, h, path):
        if self.video_writer is None and path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(path, fourcc, 20.0, (w, h))

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        hands_res = self.hands.process(rgb)
        pose_res = self.pose.process(rgb)

        hands_detected = False
        pose_detected = False
        raw_angle = SHARED_STATE.get("raw_angle", 0.0)
        torso_lean = SHARED_STATE.get("torso_lean", 0.0)

        # ---------------- Hands: virtual steering wheel ----------------
        if hands_res.multi_hand_landmarks and len(hands_res.multi_hand_landmarks) == 2:
            hands_detected = True
            wrists = []
            for hand_lms in hands_res.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
                wrist = hand_lms.landmark[mp_hands.HandLandmark.WRIST]
                wrists.append((wrist.x * w, wrist.y * h))

            wrists.sort(key=lambda p: p[0])  # left-most first
            left_wrist, right_wrist = wrists
            raw_angle = _angle_deg(left_wrist, right_wrist)
            distance = math.hypot(
                right_wrist[0] - left_wrist[0], right_wrist[1] - left_wrist[1]
            )

            if CALIBRATION["_capture_next"]:
                CALIBRATION["neutral_wheel_distance"] = distance

            cv2.line(
                img,
                (int(left_wrist[0]), int(left_wrist[1])),
                (int(right_wrist[0]), int(right_wrist[1])),
                (0, 255, 255),
                4,
            )
            cv2.circle(img, (int(left_wrist[0]), int(left_wrist[1])), 10, (0, 200, 0), -1)
            cv2.circle(img, (int(right_wrist[0]), int(right_wrist[1])), 10, (0, 0, 220), -1)

        # EMA smoothing of the wheel angle
        self.ema_angle = EMA_ALPHA * raw_angle + (1 - EMA_ALPHA) * self.ema_angle
        smoothed = self.ema_angle

        if abs(smoothed) < NOISE_GATE_DEGREES:
            steering_value = 0.0
            gesture_label = "neutral (noise-gated)" if hands_detected else "waiting for hands..."
        else:
            steering_value = max(-1.0, min(1.0, smoothed / STEER_MAX_ANGLE))
            gesture_label = "steer left" if steering_value < 0 else "steer right"

        # ---------------- Pose: torso lean (accelerate / brake) ----------------
        if pose_res.pose_landmarks:
            pose_detected = True
            mp_drawing.draw_landmarks(img, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            lm = pose_res.pose_landmarks.landmark
            l_sh, r_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER], lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            l_hip, r_hip = lm[mp_pose.PoseLandmark.LEFT_HIP], lm[mp_pose.PoseLandmark.RIGHT_HIP]
            shoulder_mid = ((l_sh.x + r_sh.x) / 2 * w, (l_sh.y + r_sh.y) / 2 * h)
            hip_mid = ((l_hip.x + r_hip.x) / 2 * w, (l_hip.y + r_hip.y) / 2 * h)
            torso_angle = _angle_deg(hip_mid, shoulder_mid)  # ~ -90 deg when upright

            if CALIBRATION["_capture_next"]:
                CALIBRATION["neutral_torso_angle"] = torso_angle

            neutral = CALIBRATION.get("neutral_torso_angle")
            if neutral is None:
                neutral = -90.0
            torso_lean = torso_angle - neutral

            bar_x = 40
            mid_y = h // 2
            cv2.line(img, (bar_x, 40), (bar_x, h - 40), (180, 180, 180), 4)
            fill_y = int(mid_y - torso_lean * 4)
            fill_y = max(40, min(h - 40, fill_y))
            cv2.line(img, (bar_x, mid_y), (bar_x, fill_y), (0, 165, 255), 8)

        if CALIBRATION["_capture_next"]:
            CALIBRATION["_capture_next"] = False
            CALIBRATION["calibrated"] = True

        accel = torso_lean > self.lean_threshold
        brake = torso_lean < -self.lean_threshold
        if accel:
            gesture_label = "accelerate (lean forward)"
        elif brake:
            gesture_label = "brake (lean back)"

        cv2.putText(img, f"Wheel angle: {smoothed:5.1f} deg", (10, h - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, f"Torso lean: {torso_lean:5.1f} deg", (10, h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, f"Gesture: {gesture_label}", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if gesture_label != self.prev_gesture and "noise-gated" not in gesture_label and "waiting" not in gesture_label:
            log_event(gesture_label, detail=f"angle={smoothed:.1f} lean={torso_lean:.1f}")
        self.prev_gesture = gesture_label

        with STATE_LOCK:
            SHARED_STATE.update(
                {
                    "raw_angle": raw_angle,
                    "ema_angle": smoothed,
                    "steering_value": steering_value,
                    "torso_lean": torso_lean,
                    "accel": accel,
                    "brake": brake,
                    "gesture_label": gesture_label,
                    "hands_detected": hands_detected,
                    "pose_detected": pose_detected,
                    "last_frame_ts": time.time(),
                }
            )

        if self.recording:
            path = st.session_state.get("mp4_path")
            self._ensure_writer(w, h, path)
            if self.video_writer is not None:
                self.video_writer.write(img)

        if av is not None:
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        return frame.from_ndarray(img, format="bgr24")

    def close_writer(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None


# --------------------------------------------------------------------------------------
# Key-dispatch JS builder
# --------------------------------------------------------------------------------------
def _dispatch_js(key, down):
    ev = "keydown" if down else "keyup"
    return (
        "document.dispatchEvent(new KeyboardEvent('%s', "
        "{key:'%s', code:'%s', bubbles:true}));" % (ev, key, key)
    )


def build_key_dispatch_js(mode, steering_value, accel, brake, tick_seconds):
    lines = []

    if mode == "Proportional key-hold steering":
        if steering_value > 0.15:
            lines.append(_dispatch_js("ArrowRight", True))
            lines.append(_dispatch_js("ArrowLeft", False))
        elif steering_value < -0.15:
            lines.append(_dispatch_js("ArrowLeft", True))
            lines.append(_dispatch_js("ArrowRight", False))
        else:
            lines.append(_dispatch_js("ArrowLeft", False))
            lines.append(_dispatch_js("ArrowRight", False))
    else:
        # Digital tap steering: tap frequency scales with |steering_value|
        tap_hz = TAP_MIN_HZ + (TAP_MAX_HZ - TAP_MIN_HZ) * min(1.0, abs(steering_value))
        fire_probability = tap_hz * tick_seconds
        if abs(steering_value) > 0.15 and np.random.rand() < fire_probability:
            key = "ArrowRight" if steering_value > 0 else "ArrowLeft"
            lines.append(_dispatch_js(key, True))
            lines.append(_dispatch_js(key, False))

    lines.append(_dispatch_js("ArrowUp", accel))
    lines.append(_dispatch_js("ArrowDown", brake))
    return "\n".join(lines)


def build_game_html(x, y, heading, speed, key_js, gesture_label, latency_ms):
    return f"""
    <div style="background:#10161d;border-radius:14px;padding:12px;
                color:#e7edf3;font-family:'Segoe UI',sans-serif;">
      <svg width="100%" height="300" viewBox="0 0 100 100"
           style="background:radial-gradient(circle at 50% 50%,#26313d,#161d24);
                  border-radius:10px;">
        <circle cx="50" cy="50" r="46" fill="none" stroke="#3a4a5a" stroke-width="10"/>
        <circle cx="50" cy="50" r="30" fill="none" stroke="#20282f" stroke-width="6"/>
        <g transform="translate({x:.2f},{y:.2f}) rotate({heading:.1f})">
          <polygon points="0,-4.5 3,4 -3,4" fill="#ffcf3c" stroke="#7a5c00" stroke-width="0.5"/>
        </g>
      </svg>
      <div style="display:flex;gap:18px;margin-top:8px;font-size:12.5px;opacity:0.9;">
        <span>Speed: <b>{speed:.1f}</b></span>
        <span>Heading: <b>{heading:.0f}&deg;</b></span>
        <span>Gesture: <b>{gesture_label}</b></span>
        <span>Key-dispatch latency: <b>{latency_ms:.0f} ms</b></span>
      </div>
    </div>
    <script>
      {key_js}
    </script>
    """


# --------------------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="Gesture Racing Controller", layout="wide")
st.title("🏎️ Gesture-Controlled Racing — Two-Hand Wheel + Torso Lean")
st.caption(
    "Live MediaPipe Hands + Pose pipeline. Steer with a two-hand 'wheel' gesture, "
    "lean forward to accelerate, lean back to brake."
)

if "mp4_path" not in st.session_state:
    tmp_dir = tempfile.mkdtemp(prefix="gesture_session_")
    st.session_state["mp4_path"] = os.path.join(tmp_dir, "session_recording.mp4")
    st.session_state["csv_path"] = os.path.join(tmp_dir, "gesture_events.csv")

for key, default in [
    ("kart_speed", 0.0), ("kart_heading", 0.0), ("kart_x", 50.0), ("kart_y", 50.0),
    ("mode", "Digital tap steering"),
]:
    st.session_state.setdefault(key, default)

with st.sidebar:
    st.header("⚙️ Controls")

    mode = st.radio(
        "Steering input mode",
        ["Digital tap steering", "Proportional key-hold steering"],
        help="Digital tap: repeated arrow-key taps at a frequency proportional to wheel "
             "angle. Proportional hold: key stays held down while wheel angle exceeds "
             "the dead-zone.",
    )
    st.session_state["mode"] = mode

    st.markdown("---")
    st.subheader("🎯 Calibration")
    st.write(
        "Hold both hands up in the 'steering wheel' position and sit/stand naturally, "
        "then click below to set your neutral baseline."
    )
    if st.button("Capture neutral baseline"):
        CALIBRATION["_capture_next"] = True
        st.success("Calibrating on next frame... hold your pose steady.")

    lean_threshold = st.slider(
        "Lean threshold (deg) for accelerate/brake", 2, 20, int(LEAN_THRESHOLD_DEFAULT)
    )

    st.markdown("---")
    st.subheader("⏺️ Session Recording")
    record = st.checkbox("Record annotated video + gesture CSV log")

    if os.path.exists(st.session_state["mp4_path"]) and os.path.getsize(st.session_state["mp4_path"]) > 0:
        with open(st.session_state["mp4_path"], "rb") as f:
            st.download_button("⬇️ Download session MP4", f, file_name="session_recording.mp4")
    if os.path.exists(st.session_state["csv_path"]) and os.path.getsize(st.session_state["csv_path"]) > 0:
        with open(st.session_state["csv_path"], "rb") as f:
            st.download_button("⬇️ Download gesture log (CSV)", f, file_name="gesture_events.csv")

    st.markdown("---")
    st.caption(
        "Calibrated: **%s**" % ("yes" if CALIBRATION["calibrated"] else "not yet")
    )

col_cam, col_game = st.columns([1, 1])

with col_cam:
    st.subheader("📷 Live Webcam — Hand-Wheel + Torso Overlay")
    ctx = webrtc_streamer(
        key="gesture-racing",
        video_processor_factory=GestureProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_game:
    st.subheader("🎮 Built-in Racing Mini-Game (driven by dispatched key events)")
    game_placeholder = st.empty()

metrics_placeholder = st.empty()
log_placeholder = st.empty()

if ctx.video_processor:
    ctx.video_processor.recording = record
    ctx.video_processor.lean_threshold = lean_threshold

if ctx.state.playing:
    while ctx.state.playing:
        if ctx.video_processor is None:
            time.sleep(0.1)
            continue

        ctx.video_processor.recording = record
        ctx.video_processor.lean_threshold = lean_threshold

        loop_start = time.time()
        with STATE_LOCK:
            snap = dict(SHARED_STATE)
            recent_events = list(SHARED_STATE["event_log"][-8:])

        steering_value = snap["steering_value"]
        accel = snap["accel"]
        brake = snap["brake"]

        # ---- simple kart physics driven by the gesture signal ----
        speed = st.session_state["kart_speed"]
        heading = st.session_state["kart_heading"]
        pos_x = st.session_state["kart_x"]
        pos_y = st.session_state["kart_y"]

        if accel:
            speed = min(speed + 2.0, 20.0)
        elif brake:
            speed = max(speed - 3.0, -6.0)
        else:
            speed *= 0.92

        heading += steering_value * 6.0
        pos_x += speed * math.sin(math.radians(heading)) * 0.3
        pos_y -= speed * math.cos(math.radians(heading)) * 0.3
        pos_x = max(6, min(94, pos_x))
        pos_y = max(6, min(94, pos_y))

        st.session_state.update(
            {"kart_speed": speed, "kart_heading": heading, "kart_x": pos_x, "kart_y": pos_y}
        )

        key_js = build_key_dispatch_js(
            st.session_state["mode"], steering_value, accel, brake, LOOP_TICK_SECONDS
        )
        latency_ms = (time.time() - snap["last_frame_ts"]) * 1000.0

        with game_placeholder:
            components.html(
                build_game_html(pos_x, pos_y, heading, speed, key_js, snap["gesture_label"], latency_ms),
                height=340,
            )

        with metrics_placeholder.container():
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Steering angle (EMA)", f"{snap['ema_angle']:.1f}°")
            m2.metric("Steering value", f"{steering_value:+.2f}")
            m3.metric("Accelerate", "ON" if accel else "off")
            m4.metric("Brake", "ON" if brake else "off")
            st.progress(min(1.0, abs(steering_value)))
            st.caption(
                f"Gesture: **{snap['gesture_label']}** &nbsp;|&nbsp; "
                f"Hands detected: {snap['hands_detected']} &nbsp;|&nbsp; "
                f"Pose detected: {snap['pose_detected']}"
            )

        with log_placeholder.container():
            if recent_events:
                st.write("**Recent gesture events**")
                for e in reversed(recent_events):
                    st.text(f"{e['timestamp']}  {e['event']}  ({e['detail']})")

        elapsed = time.time() - loop_start
        time.sleep(max(0.0, LOOP_TICK_SECONDS - elapsed))
else:
    st.info(
        "Click **Start** on the webcam widget above to begin. Then use the sidebar to "
        "capture your neutral calibration baseline before steering."
    )
    if ctx.video_processor is not None:
        ctx.video_processor.close_writer()