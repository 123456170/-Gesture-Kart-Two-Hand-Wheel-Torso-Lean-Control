import os
import time
import json
import csv
import math
import threading
import tempfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import cv2
import av
import mediapipe as mp

import streamlit as st
import streamlit.components.v1 as components

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

try:
    from streamlit_webrtc import WebRtcMode
    WEBRTC_MODE = WebRtcMode.SENDRECV
except Exception:
    WEBRTC_MODE = "SENDRECV"

try:
    from streamlit_webrtc.models import MediaStreamConstraints
    MEDIA_CONSTRAINTS = MediaStreamConstraints(video=True, audio=False)
except Exception:
    MEDIA_CONSTRAINTS = {"video": True, "audio": False}

st.set_page_config(
    page_title="Gesture Kart - Hands + Torso Racing Control",
    page_icon="🏎️",
    layout="wide",
)


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, float(value)))


def wrap_angle(angle_deg):
    return ((float(angle_deg) + 180.0) % 360.0) - 180.0


class SharedState:
    """
    Thread-safe shared state between:
    - MediaPipe frame processor thread
    - Streamlit UI thread
    - local JSON HTTP server consumed by the embedded browser game
    """

    def __init__(self):
        self.lock = threading.RLock()

        # Input / control modes
        self.mode = "tap"  # tap | hold
        self.demo_assist = True

        # Calibration
        self.calibrating = True
        self.calib_start = time.time()
        self.calibrated = False
        self.calib_hand_samples = []
        self.calib_torso_samples = []

        # Neutral baselines
        self.neutral_angle = 0.0
        self.neutral_wrist_dist = 0.0
        self.neutral_torso = 0.0

        # Live steering / torso state
        self.raw_angle = 0.0
        self.smooth_angle = 0.0
        self.steer = 0.0
        self.torso_lean = 0.0
        self.accel = False
        self.brake = False

        # Detection status
        self.hands_count = 0
        self.pose_ok = False
        self.gesture = "Initializing"
        self.last_gesture = "Initializing"
        self.confidence = 0.0

        # Performance
        self.latency_ms = 0.0
        self.fps = 0.0
        self.frame_count = 0
        self.fps_time = time.time()
        self.last_process_time = 0.0

        # Browser key dispatch
        self.tap_events = []
        self.hold = {
            "ArrowLeft": False,
            "ArrowRight": False,
            "ArrowUp": False,
            "ArrowDown": False,
        }
        self.last_left_tap = 0.0
        self.last_right_tap = 0.0

        # Edge-case / noise rejection
        self.angle_history = deque(maxlen=18)

        # Recording
        self.record = False
        self.writer = None
        self.record_size = None
        self.video_path = os.path.join(tempfile.gettempdir(), "gesture_kart_live_demo.mp4")
        self.csv_path = os.path.join(tempfile.gettempdir(), "gesture_kart_event_log.csv")

        # Tunables
        self.deadzone = 4.0
        self.max_steer_angle = 35.0
        self.accel_threshold = 6.0
        self.brake_threshold = 6.0
        self.invert_steer = False
        self.invert_lean = False
        self.ema_alpha = 0.25
        self.min_tap_hz = 2.0
        self.max_tap_hz = 10.0

    def log_event(self, event, detail="", steer=None, lean=None, confidence=None):
        try:
            with self.lock:
                write_header = not os.path.exists(self.csv_path)
                with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if write_header:
                        writer.writerow([
                            "timestamp_unix",
                            "timestamp_local",
                            "event",
                            "detail",
                            "steer_value",
                            "smoothed_wheel_angle_deg",
                            "torso_lean_deg",
                            "confidence",
                        ])
                    writer.writerow([
                        time.time(),
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        event,
                        detail,
                        self.steer if steer is None else steer,
                        self.smooth_angle,
                        self.torso_lean if lean is None else lean,
                        self.confidence if confidence is None else confidence,
                    ])
        except Exception:
            # Logging should never crash the live pipeline.
            pass


if hasattr(st, "cache_resource"):
    @st.cache_resource
    def get_shared_state():
        return SharedState()

    @st.cache_resource
    def start_state_server():
        for port in range(8765, 8795):
            try:
                server = ThreadingHTTPServer(("127.0.0.1", port), StateHandler)
                server.daemon_threads = True
                threading.Thread(target=server.serve_forever, daemon=True).start()
                return port
            except OSError:
                continue
        return None
else:
    # Fallback for very old Streamlit versions.
    def get_shared_state():
        if "shared_state" not in st.session_state:
            st.session_state.shared_state = SharedState()
        return st.session_state.shared_state

    def start_state_server():
        if "_gesture_server_port" not in globals():
            port = None
            for candidate in range(8765, 8795):
                try:
                    server = ThreadingHTTPServer(("127.0.0.1", candidate), StateHandler)
                    server.daemon_threads = True
                    threading.Thread(target=server.serve_forever, daemon=True).start()
                    port = candidate
                    break
                except OSError:
                    continue
            globals()["_gesture_server_port"] = port
        return globals()["_gesture_server_port"]


STATE = get_shared_state()


class StateHandler(BaseHTTPRequestHandler):
    state = None

    def _set_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")

    def do_OPTIONS(self):
        self.send_response(204)
        self._set_cors()
        self.end_headers()

    def do_GET(self):
        if self.path != "/state":
            self.send_response(404)
            self._set_cors()
            self.end_headers()
            return

        state = self.__class__.state
        if state is None:
            self.send_response(503)
            self._set_cors()
            self.end_headers()
            return

        with state.lock:
            active = (time.time() - state.last_process_time) < 1.0
            taps = state.tap_events[:]
            state.tap_events.clear()

            payload = {
                "active": active,
                "demo_assist": bool(state.demo_assist),
                "mode": state.mode,
                "steer": float(state.steer),
                "raw_angle": float(state.raw_angle),
                "smooth_angle": float(state.smooth_angle),
                "torso_lean": float(state.torso_lean),
                "accel": bool(state.accel),
                "brake": bool(state.brake),
                "hands_count": int(state.hands_count),
                "pose_ok": bool(state.pose_ok),
                "gesture": state.gesture,
                "confidence": float(state.confidence),
                "latency_ms": float(state.latency_ms),
                "calibrated": bool(state.calibrated),
                "calibrating": bool(state.calibrating),
                "hold": dict(state.hold),
                "tap_events": taps,
            }

        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self._set_cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args, **kwargs):
        # Keep console quiet.
        pass


StateHandler.state = STATE
SERVER_PORT = start_state_server()


def compute_torso_lean(pose_results):
    """
    Returns (lean_angle_degrees, pose_ok).
    Uses MediaPipe Pose world landmarks when available: shoulder midpoint vs hip midpoint.
    Positive/negative direction can be inverted in the sidebar.
    """
    try:
        world = getattr(pose_results, "pose_world_landmarks", None)
        if world is not None and len(world.landmark) > 24:
            lm = world.landmark

            shoulder_x = (lm[11].x + lm[12].x) * 0.5
            shoulder_y = (lm[11].y + lm[12].y) * 0.5
            shoulder_z = (lm[11].z + lm[12].z) * 0.5

            hip_x = (lm[23].x + lm[24].x) * 0.5
            hip_y = (lm[23].y + lm[24].y) * 0.5
            hip_z = (lm[23].z + lm[24].z) * 0.5

            vx = shoulder_x - hip_x
            vy = shoulder_y - hip_y
            vz = shoulder_z - hip_z

            norm = math.sqrt(vx * vx + vy * vy + vz * vz)
            if norm < 1e-6:
                return 0.0, False

            # Sagittal-plane pitch: torso vector relative to vertical.
            pitch = math.degrees(math.atan2(vz, vy))
            return float(pitch), True

        if pose_results.pose_landmarks is not None and len(pose_results.pose_landmarks.landmark) > 24:
            lm = pose_results.pose_landmarks.landmark

            shoulder_z = (lm[11].z + lm[12].z) * 0.5
            hip_z = (lm[23].z + lm[24].z) * 0.5

            shoulder_y = (lm[11].y + lm[12].y) * 0.5
            hip_y = (lm[23].y + lm[24].y) * 0.5

            torso_len = max(1e-3, abs(hip_y - shoulder_y))
            pitch = math.degrees(math.atan2(shoulder_z - hip_z, torso_len))
            return float(pitch), True

    except Exception:
        return 0.0, False

    return 0.0, False


def draw_badge(img, text, color=(0, 220, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    x0, y0 = 12, 12
    x1, y1 = x0 + tw + 18, y0 + th + baseline + 14

    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
    cv2.putText(
        img,
        text,
        (x0 + 9, y0 + th + 7),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_steering_gauge(img, steer, angle_deg):
    h, w = img.shape[:2]
    x, y, bw, bh = 20, h - 88, 360, 26

    cv2.rectangle(img, (x, y), (x + bw, y + bh), (35, 35, 35), -1)
    cv2.rectangle(img, (x, y), (x + bw, y + bh), (255, 255, 255), 1)

    center = x + bw // 2
    cv2.line(img, (center, y), (center, y + bh), (210, 210, 210), 1)

    pos = int(center + clamp(steer, -1.0, 1.0) * (bw // 2 - 4))
    color = (0, 255, 120) if abs(steer) < 0.05 else (0, 200, 255)
    cv2.circle(img, (pos, y + bh // 2), 8, color, -1)

    label = f"Steering angle {angle_deg:+.1f} deg | steer value {steer:+.2f}"
    cv2.putText(
        img,
        label,
        (x, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_torso_bar(img, lean_deg, accel, brake):
    h, w = img.shape[:2]
    x, y, bw, bh = w - 270, 26, 250, 24

    cv2.rectangle(img, (x, y), (x + bw, y + bh), (35, 35, 35), -1)
    cv2.rectangle(img, (x, y), (x + bw, y + bh), (255, 255, 255), 1)

    center = x + bw // 2
    cv2.line(img, (center, y), (center, y + bh), (210, 210, 210), 1)

    clamped = clamp(lean_deg, -25.0, 25.0)
    pos = int(center + (clamped / 25.0) * (bw // 2 - 3))

    if accel:
        color = (0, 255, 120)
        label = "Torso: ACCELERATE"
    elif brake:
        color = (80, 80, 255)
        label = "Torso: BRAKE"
    else:
        color = (0, 200, 255)
        label = f"Torso lean {lean_deg:+.1f} deg"

    cv2.circle(img, (pos, y + bh // 2), 7, color, -1)
    cv2.putText(
        img,
        label,
        (x, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


class GestureProcessor(VideoProcessorBase):
    """
    Processes each webcam frame with MediaPipe Hands and MediaPipe Pose.
    Hands and Pose are submitted to a two-worker thread pool so they run in parallel.
    """

    def __init__(self):
        super().__init__()

        self.hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose = mp.solutions.pose.Pose(
            model_complexity=1,
            enable_segmentation=False,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.draw = mp.solutions.drawing_utils
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.smooth_initialized = False
        self.calib_started = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        t0 = time.time()

        try:
            img = frame.to_ndarray(format="bgr24")
        except Exception:
            return frame

        try:
            # Mirror for a natural driving-camera view.
            img = cv2.flip(img, 1)
            h, w = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Run Hands and Pose in parallel on the same frame.
            hands_future = self.executor.submit(self.hands.process, rgb)
            pose_future = self.executor.submit(self.pose.process, rgb)

            try:
                hands_res = hands_future.result(timeout=1.0)
                pose_res = pose_future.result(timeout=1.0)
            except Exception:
                hands_res = None
                pose_res = None

            hands_count = 0
            hand_ok = False
            raw_wheel_angle = 0.0
            wrist_dist = 0.0

            # Hands: draw and compute wrist line angle.
            if hands_res is not None and hands_res.multi_hand_landmarks:
                hands_count = len(hands_res.multi_hand_landmarks)

                for hand_landmarks in hands_res.multi_hand_landmarks:
                    self.draw.draw_landmarks(
                        img,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                    )

                if hands_count >= 2:
                    wrist_a = hands_res.multi_hand_landmarks[0].landmark[0]
                    wrist_b = hands_res.multi_hand_landmarks[1].landmark[0]

                    ax, ay = wrist_a.x * w, wrist_a.y * h
                    bx, by = wrist_b.x * w, wrist_b.y * h

                    if ax <= bx:
                        lx, ly, rx, ry = ax, ay, bx, by
                    else:
                        lx, ly, rx, ry = bx, by, ax, ay

                    dx = rx - lx
                    dy = ry - ly
                    wrist_dist = math.hypot(dx, dy)
                    raw_wheel_angle = math.degrees(math.atan2(dy, dx))
                    hand_ok = True

            # Pose: draw and compute torso lean.
            torso_raw = 0.0
            pose_ok = False

            if pose_res is not None and pose_res.pose_landmarks:
                self.draw.draw_landmarks(
                    img,
                    pose_res.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS,
                )

            if pose_res is not None:
                torso_raw, pose_ok = compute_torso_lean(pose_res)

            rel_angle = 0.0
            steer = 0.0
            lean_rel = 0.0
            accel = False
            brake = False
            gesture = "Initializing"
            confidence = 0.0
            ambiguous = False
            dist_ok = True
            calibrating_local = False
            calib_remaining = 0.0

            with STATE.lock:
                now = time.time()
                STATE.last_process_time = now

                STATE.frame_count += 1
                if now - STATE.fps_time >= 1.0:
                    STATE.fps = STATE.frame_count / max(1e-6, now - STATE.fps_time)
                    STATE.frame_count = 0
                    STATE.fps_time = now

                STATE.hands_count = hands_count
                STATE.pose_ok = pose_ok

                # If the app was opened but webcam START was pressed later,
                # restart the 3-second calibration clock at first real frame.
                if STATE.calibrating and not self.calib_started:
                    STATE.calib_start = now
                    self.calib_started = True

                if not STATE.calibrating:
                    self.calib_started = False

                # Startup / manual calibration.
                if STATE.calibrating:
                    if hand_ok:
                        STATE.calib_hand_samples.append((raw_wheel_angle, wrist_dist))
                    if pose_ok:
                        STATE.calib_torso_samples.append(torso_raw)

                    elapsed = now - STATE.calib_start
                    calib_remaining = max(0.0, 3.0 - elapsed)

                    if elapsed >= 3.0:
                        if len(STATE.calib_hand_samples) >= 8:
                            angles = np.array([a for a, _ in STATE.calib_hand_samples], dtype=float)
                            dists = np.array([d for _, d in STATE.calib_hand_samples], dtype=float)
                            STATE.neutral_angle = float(np.median(angles))
                            STATE.neutral_wrist_dist = float(np.median(dists))
                        elif hand_ok:
                            STATE.neutral_angle = raw_wheel_angle
                            STATE.neutral_wrist_dist = wrist_dist

                        if len(STATE.calib_torso_samples) >= 8:
                            torso_samples = np.array(STATE.calib_torso_samples, dtype=float)
                            STATE.neutral_torso = float(np.median(torso_samples))
                        elif pose_ok:
                            STATE.neutral_torso = torso_raw
                        else:
                            STATE.neutral_torso = 0.0

                        STATE.calibrated = True
                        STATE.calibrating = False
                        STATE.calib_hand_samples = []
                        STATE.calib_torso_samples = []
                        self.calib_started = False

                        STATE.log_event(
                            "calibration",
                            f"neutral_angle={STATE.neutral_angle:.2f}, "
                            f"neutral_wrist_dist={STATE.neutral_wrist_dist:.1f}, "
                            f"neutral_torso={STATE.neutral_torso:.2f}",
                        )

                calibrating_local = STATE.calibrating

                # Relative wheel angle compared with calibrated neutral.
                if hand_ok:
                    rel_angle = wrap_angle(raw_wheel_angle - STATE.neutral_angle)
                else:
                    rel_angle = 0.0

                STATE.angle_history.append(rel_angle if hand_ok else 0.0)

                # EMA smoothing.
                if not self.smooth_initialized:
                    STATE.smooth_angle = rel_angle
                    self.smooth_initialized = True
                else:
                    alpha = clamp(STATE.ema_alpha, 0.05, 0.95)
                    if hand_ok:
                        STATE.smooth_angle = alpha * rel_angle + (1.0 - alpha) * STATE.smooth_angle
                    else:
                        # Decay toward neutral when hands are lost.
                        STATE.smooth_angle = 0.8 * STATE.smooth_angle

                STATE.raw_angle = rel_angle

                # Validate wheel-hand distance against calibration.
                dist_ok = True
                if STATE.calibrated and STATE.neutral_wrist_dist > 1.0 and wrist_dist > 1.0:
                    ratio = wrist_dist / STATE.neutral_wrist_dist
                    dist_ok = 0.45 < ratio < 2.3

                steering_valid = hand_ok and dist_ok

                # Map smoothed angle to continuous steering value.
                dead = float(max(0.0, STATE.deadzone))
                max_ang = float(max(dead + 1.0, STATE.max_steer_angle))

                if steering_valid and abs(STATE.smooth_angle) > dead:
                    steer = math.copysign(
                        min(1.0, (abs(STATE.smooth_angle) - dead) / (max_ang - dead)),
                        STATE.smooth_angle,
                    )
                else:
                    steer = 0.0

                if STATE.invert_steer:
                    steer = -steer

                # Torso lean relative to calibrated neutral.
                if pose_ok:
                    lean_rel = torso_raw - STATE.neutral_torso
                    if STATE.invert_lean:
                        lean_rel = -lean_rel
                else:
                    lean_rel = 0.0

                STATE.torso_lean = lean_rel

                # Accelerate / brake hold states.
                if (not STATE.calibrating) and pose_ok:
                    accel = lean_rel >= STATE.accel_threshold
                    brake = lean_rel <= -STATE.brake_threshold
                    if accel and brake:
                        brake = False
                else:
                    accel = False
                    brake = False

                # Edge-case / ambiguous wobble rejection.
                angle_std = 0.0
                if len(STATE.angle_history) >= 8:
                    angle_std = float(np.std(list(STATE.angle_history)))

                ambiguous = steering_valid and abs(steer) < 0.05 and angle_std > 2.0

                steer_conf = 0.0
                if steering_valid:
                    steer_conf = 0.9 if STATE.calibrated and dist_ok else 0.65

                pose_conf = 0.8 if pose_ok else 0.0
                confidence = max(steer_conf, pose_conf)

                if ambiguous:
                    confidence = min(confidence, 0.35)

                # Gesture label.
                if STATE.calibrating:
                    gesture = f"Calibrating: hold steady ({calib_remaining:.1f}s)"
                elif accel:
                    gesture = "Accelerate"
                elif brake:
                    gesture = "Brake"
                elif not hand_ok:
                    gesture = "Show both hands to steer"
                elif not dist_ok:
                    gesture = "Hold wheel hands at calibrated distance"
                elif steer <= -0.10:
                    gesture = "Steer left"
                elif steer >= 0.10:
                    gesture = "Steer right"
                elif ambiguous:
                    gesture = "Ignored: ambiguous wobble"
                else:
                    gesture = "Neutral"

                STATE.steer = steer
                STATE.accel = accel
                STATE.brake = brake
                STATE.gesture = gesture
                STATE.confidence = confidence
                STATE.latency_ms = (time.time() - t0) * 1000.0

                if gesture != STATE.last_gesture:
                    STATE.log_event("gesture", gesture, steer, lean_rel, confidence)
                    STATE.last_gesture = gesture

                # Hold keys: accelerate / brake are held in both modes.
                prev_hold = STATE.hold.copy()
                STATE.hold["ArrowUp"] = bool(accel)
                STATE.hold["ArrowDown"] = bool(brake)

                allow_steering = (
                    (not STATE.calibrating)
                    and steering_valid
                    and confidence >= 0.45
                )

                if STATE.mode == "hold":
                    STATE.hold["ArrowLeft"] = bool(allow_steering and steer <= -0.05)
                    STATE.hold["ArrowRight"] = bool(allow_steering and steer >= 0.05)
                else:
                    STATE.hold["ArrowLeft"] = False
                    STATE.hold["ArrowRight"] = False

                for key, down in STATE.hold.items():
                    if down != prev_hold.get(key, False):
                        STATE.log_event(
                            "key_down" if down else "key_up",
                            key,
                            steer,
                            lean_rel,
                            confidence,
                        )

                # Digital tap steering: sharper angle => faster repeated taps.
                if STATE.mode == "tap" and allow_steering:
                    if len(STATE.tap_events) > 500:
                        STATE.tap_events = STATE.tap_events[-200:]

                    mag = abs(steer)
                    if mag > 0.05:
                        freq = STATE.min_tap_hz + (STATE.max_tap_hz - STATE.min_tap_hz) * mag
                        freq = clamp(freq, 0.5, 25.0)
                        interval = 1.0 / max(0.5, freq)

                        if steer < 0 and now - STATE.last_left_tap >= interval:
                            STATE.tap_events.append({"key": "ArrowLeft", "ts": now})
                            STATE.last_left_tap = now
                            STATE.log_event("key_tap", "ArrowLeft", steer, lean_rel, confidence)

                        elif steer > 0 and now - STATE.last_right_tap >= interval:
                            STATE.tap_events.append({"key": "ArrowRight", "ts": now})
                            STATE.last_right_tap = now
                            STATE.log_event("key_tap", "ArrowRight", steer, lean_rel, confidence)

            # Overlays: calibration box.
            if calibrating_local:
                cv2.rectangle(
                    img,
                    (w // 2 - 180, h // 2 - 120),
                    (w // 2 + 180, h // 2 + 120),
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    img,
                    f"CALIBRATION: hold hands level here ({calib_remaining:.1f}s)",
                    (w // 2 - 230, h // 2 - 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    img,
                    "Keep both wrists visible and sit in neutral posture",
                    (w // 2 - 210, h // 2 + 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Overlays: gesture badge.
            badge_color = (0, 220, 0)
            if "Ignored" in gesture:
                badge_color = (0, 200, 255)
            elif "Calibrating" in gesture or "Show" in gesture or "distance" in gesture:
                badge_color = (0, 220, 255)
            elif gesture in ("Accelerate", "Steer left", "Steer right"):
                badge_color = (0, 255, 120)
            elif gesture == "Brake":
                badge_color = (80, 80, 255)

            badge_text = f"{time.strftime('%H:%M:%S')} | {gesture} | conf {confidence:.2f}"
            draw_badge(img, badge_text, badge_color)

            # Overlays: steering gauge and torso bar.
            draw_steering_gauge(img, steer, STATE.smooth_angle)
            draw_torso_bar(img, lean_rel, accel, brake)

            # Overlays: latency / FPS / status.
            status = (
                f"FPS {STATE.fps:.1f} | latency {STATE.latency_ms:.0f} ms | "
                f"mode {STATE.mode} | hands {hands_count} | pose {'OK' if pose_ok else '--'}"
            )
            cv2.putText(
                img,
                status,
                (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Overlays: recording indicator.
            if STATE.record:
                cv2.circle(img, (w - 26, 26), 8, (0, 0, 255), -1)
                cv2.putText(
                    img,
                    "REC",
                    (w - 64, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Recording: write annotated frame.
            with STATE.lock:
                if STATE.record:
                    try:
                        if STATE.writer is None or STATE.record_size != (w, h):
                            if STATE.writer is not None:
                                STATE.writer.release()

                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            STATE.writer = cv2.VideoWriter(
                                STATE.video_path,
                                fourcc,
                                24.0,
                                (w, h),
                            )
                            STATE.record_size = (w, h)

                        if STATE.writer is not None and STATE.writer.isOpened():
                            STATE.writer.write(img)
                        else:
                            STATE.writer = None
                    except Exception:
                        STATE.writer = None
                else:
                    if STATE.writer is not None:
                        try:
                            STATE.writer.release()
                        except Exception:
                            pass
                        STATE.writer = None
                        STATE.record_size = None

            return av.VideoFrame.from_ndarray(img, format="bgr24")

        except Exception as exc:
            try:
                cv2.putText(
                    img,
                    f"Pipeline error: {exc}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            except Exception:
                return frame


GAME_HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<style>
  body {
    margin: 0;
    background: #0b0f14;
    color: #eaf2f7;
    font-family: Arial, Helvetica, sans-serif;
  }
  .wrap {
    padding: 10px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  canvas {
    width: 100%;
    max-width: 760px;
    border-radius: 12px;
    background: #111823;
    box-shadow: 0 0 0 1px rgba(255,255,255,0.08);
  }
  #hud {
    font-size: 13px;
    line-height: 1.5;
    background: rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 10px;
    max-width: 760px;
  }
  .hint {
    font-size: 12px;
    opacity: 0.85;
    max-width: 760px;
  }
</style>
</head>
<body>
<div class="wrap">
  <canvas id="game" width="760" height="430"></canvas>
  <div id="hud">Waiting for live gesture state...</div>
  <div class="hint">
    This embedded game listens for injected KeyboardEvent arrows. You can also click inside this panel
    and drive with physical arrow keys for comparison. If the live webcam pipeline is not active yet,
    Demo Assist automatically animates the game so the dashboard is never blank.
  </div>
</div>

<script>
const PORT = "__PORT__";
const STATE_URL = "http://127.0.0.1:" + PORT + "/state";

const canvas = document.getElementById("game");
const ctx = canvas.getContext("2d");

const gameKeys = ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"];
let keys = {};

function press(key) {
  if (!keys[key]) {
    window.dispatchEvent(new KeyboardEvent("keydown", {
      key: key,
      bubbles: true,
      cancelable: true
    }));
    keys[key] = true;
  }
}

function release(key) {
  if (keys[key]) {
    window.dispatchEvent(new KeyboardEvent("keyup", {
      key: key,
      bubbles: true,
      cancelable: true
    }));
    keys[key] = false;
  }
}

function setKey(key, down) {
  if (down) press(key);
  else release(key);
}

function tapKey(key) {
  window.dispatchEvent(new KeyboardEvent("keydown", {
    key: key,
    bubbles: true,
    cancelable: true
  }));

  setTimeout(function() {
    window.dispatchEvent(new KeyboardEvent("keyup", {
      key: key,
      bubbles: true,
      cancelable: true
    }));
  }, 70);
}

window.addEventListener("keydown", function(e) {
  if (gameKeys.includes(e.key)) {
    keys[e.key] = true;
    e.preventDefault();
  }
});

window.addEventListener("keyup", function(e) {
  if (gameKeys.includes(e.key)) {
    keys[e.key] = false;
    e.preventDefault();
  }
});

let lastData = null;
let failCount = 0;

function simulate(data) {
  if (!data) {
    data = {};
    lastData = data;
  }

  const t = performance.now() / 1000.0;
  const steer = Math.sin(t * 0.9) * 0.72 + Math.sin(t * 2.7) * 0.08;
  const lean = Math.sin(t * 0.42) * 9.0;

  setKey("ArrowLeft", steer < -0.18);
  setKey("ArrowRight", steer > 0.18);
  setKey("ArrowUp", lean > 2.5);
  setKey("ArrowDown", lean < -5.0);

  data.simulated = true;
  data.active = false;
  data.mode = data.mode || "sim";
  data.steer = steer;
  data.smooth_angle = steer * 30.0;
  data.torso_lean = lean;
  data.accel = lean > 2.5;
  data.brake = lean < -5.0;
  data.gesture = "DEMO ASSIST / SIMULATED";
  data.confidence = 1.0;
  data.latency_ms = 12.0;
  data.hands_count = data.hands_count || 0;
  data.pose_ok = true;
  data.calibrated = true;
  data.calibrating = false;
}

async function pollState() {
  try {
    const res = await fetch(STATE_URL, { cache: "no-store" });
    const data = await res.json();

    failCount = 0;
    lastData = data;

    const shouldSimulate =
      (!data.active) ||
      (data.demo_assist && data.hands_count < 2 && !data.calibrating);

    if (shouldSimulate) {
      simulate(data);
    } else {
      data.simulated = false;

      const hold = data.hold || {};
      gameKeys.forEach(function(k) {
        setKey(k, Boolean(hold[k]));
      });

      const taps = data.tap_events || [];
      taps.forEach(function(ev) {
        if (ev && ev.key) tapKey(ev.key);
      });
    }

    updateHud();
  } catch (err) {
    failCount += 1;
    if (failCount > 8) {
      simulate(null);
      updateHud();
    }
  }

  setTimeout(pollState, 33);
}

function fmt(v, digits) {
  if (v === undefined || v === null || isNaN(v)) return "--";
  return Number(v).toFixed(digits === undefined ? 1 : digits);
}

function txt(v) {
  if (v === undefined || v === null) return "--";
  return v;
}

function updateHud() {
  if (!lastData) return;

  const d = lastData;
  const liveBadge = d.simulated
    ? '<span style="color:#ffd166;font-weight:bold;">SIM / DEMO ASSIST</span>'
    : '<span style="color:#06d6a0;font-weight:bold;">LIVE</span>';

  document.getElementById("hud").innerHTML =
    liveBadge +
    " | Mode: <b>" + txt(d.mode) + "</b>" +
    " | Gesture: <b>" + txt(d.gesture) + "</b>" +
    " | Confidence: <b>" + fmt(d.confidence, 2) + "</b><br>" +

    "Steering: <b>" + fmt(d.steer, 2) + "</b>" +
    " | Wheel angle: <b>" + fmt(d.smooth_angle, 1) + "°</b>" +
    " | Torso lean: <b>" + fmt(d.torso_lean, 1) + "°</b>" +
    " | Accel: <b>" + (d.accel ? "YES" : "no") + "</b>" +
    " | Brake: <b>" + (d.brake ? "YES" : "no") + "</b><br>" +

    "Hands: <b>" + txt(d.hands_count) + "</b>" +
    " | Pose: <b>" + (d.pose_ok ? "OK" : "--") + "</b>" +
    " | Calibrated: <b>" + (d.calibrated ? "YES" : "NO") + "</b>" +
    " | Calibrating: <b>" + (d.calibrating ? "YES" : "NO") + "</b>" +
    " | Latency: <b>" + fmt(d.latency_ms, 0) + " ms</b>";
}

let kartX = canvas.width / 2;
let speed = 0;
let roadY = 0;
const maxSpeed = 360;
let lastFrame = performance.now();

function update(dt) {
  const roadLeft = 150;
  const roadRight = canvas.width - 150;

  if (keys["ArrowUp"]) speed += 220 * dt;
  else speed -= 55 * dt;

  if (keys["ArrowDown"]) speed -= 420 * dt;

  speed = Math.max(0, Math.min(maxSpeed, speed));

  const steerPower = 190 + speed * 0.65;

  if (keys["ArrowLeft"]) kartX -= steerPower * dt;
  if (keys["ArrowRight"]) kartX += steerPower * dt;

  kartX = Math.max(40, Math.min(canvas.width - 40, kartX));

  if (kartX < roadLeft + 25 || kartX > roadRight - 25) {
    speed *= Math.max(0.0, 1.0 - 0.9 * dt);
  }

  roadY += speed * dt;
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Grass
  ctx.fillStyle = "#123524";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Road
  const roadLeft = 150;
  const roadRight = canvas.width - 150;
  ctx.fillStyle = "#2b2f36";
  ctx.fillRect(roadLeft, 0, roadRight - roadLeft, canvas.height);

  // Road edges
  ctx.strokeStyle = "#f2f2f2";
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(roadLeft, 0);
  ctx.lineTo(roadLeft, canvas.height);
  ctx.moveTo(roadRight, 0);
  ctx.lineTo(roadRight, canvas.height);
  ctx.stroke();

  // Dashed centerline
  ctx.strokeStyle = "#ffd166";
  ctx.lineWidth = 5;
  ctx.setLineDash([24, 26]);
  ctx.lineDashOffset = -roadY;
  ctx.beginPath();
  ctx.moveTo(canvas.width / 2, 0);
  ctx.lineTo(canvas.width / 2, canvas.height);
  ctx.stroke();
  ctx.setLineDash([]);

  // Kart
  ctx.save();
  ctx.translate(kartX, canvas.height - 85);

  ctx.fillStyle = "#ff4d4d";
  ctx.beginPath();
  ctx.moveTo(0, -26);
  ctx.lineTo(18, 22);
  ctx.lineTo(-18, 22);
  ctx.closePath();
  ctx.fill();

  ctx.fillStyle = "#111";
  ctx.fillRect(-24, 8, 10, 18);
  ctx.fillRect(14, 8, 10, 18);

  ctx.fillStyle = "#dff5ff";
  ctx.beginPath();
  ctx.arc(0, -2, 7, 0, Math.PI * 2);
  ctx.fill();

  ctx.restore();

  // Speed bar
  ctx.fillStyle = "rgba(255,255,255,0.14)";
  ctx.fillRect(20, 20, 220, 16);
  ctx.fillStyle = speed > 260 ? "#ff6b6b" : "#06d6a0";
  ctx.fillRect(20, 20, 220 * (speed / maxSpeed), 16);
  ctx.strokeStyle = "rgba(255,255,255,0.4)";
  ctx.strokeRect(20, 20, 220, 16);

  ctx.fillStyle = "#ffffff";
  ctx.font = "12px Arial";
  ctx.fillText("Speed: " + Math.round(speed), 20, 52);

  if (lastData && lastData.simulated) {
    ctx.fillStyle = "rgba(255,209,102,0.9)";
    ctx.font = "bold 14px Arial";
    ctx.fillText("DEMO ASSIST: start webcam and show both hands to take over", 20, canvas.height - 18);
  }
}

function loop(now) {
  const dt = Math.min(0.05, (now - lastFrame) / 1000.0);
  lastFrame = now;

  update(dt);
  draw();

  requestAnimationFrame(loop);
}

pollState();
requestAnimationFrame(loop);
</script>
</body>
</html>
"""


# ----------------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------------

st.sidebar.title("Gesture Kart Controls")

mode_options = ("Digital tap steering", "Proportional key-hold steering")
mode_index = 0 if STATE.mode == "tap" else 1

mode_choice = st.sidebar.radio(
    "Steering mode",
    mode_options,
    index=mode_index,
    key="steering_mode_selector",
    help=(
        "Digital tap steering repeats left/right arrow taps at a rate proportional to wheel angle. "
        "Proportional key-hold steering holds the arrow key while the wheel angle is beyond deadzone."
    ),
)

STATE.mode = "tap" if mode_choice == mode_options[0] else "hold"

STATE.demo_assist = st.sidebar.checkbox(
    "Demo assist (auto-simulate until hands lock)",
    value=STATE.demo_assist,
    key="demo_assist_toggle",
    help="Keeps the embedded game alive with simulated driving until your real webcam gestures take over.",
)

STATE.invert_steer = st.sidebar.checkbox(
    "Invert steering direction",
    value=STATE.invert_steer,
    key="invert_steer_toggle",
)

STATE.invert_lean = st.sidebar.checkbox(
    "Invert forward/backward lean",
    value=STATE.invert_lean,
    key="invert_lean_toggle",
)

STATE.deadzone = st.sidebar.slider(
    "Steering deadzone (deg)",
    0.0,
    15.0,
    float(STATE.deadzone),
    0.5,
    key="deadzone_slider",
    help="Small wrist-line rotations inside this deadzone are ignored as noise.",
)

STATE.max_steer_angle = st.sidebar.slider(
    "Max steering angle (deg)",
    10.0,
    90.0,
    float(STATE.max_steer_angle),
    1.0,
    key="max_steer_slider",
)

STATE.accel_threshold = st.sidebar.slider(
    "Accelerate lean threshold (deg)",
    1.0,
    25.0,
    float(STATE.accel_threshold),
    0.5,
    key="accel_threshold_slider",
)

STATE.brake_threshold = st.sidebar.slider(
    "Brake lean threshold (deg)",
    1.0,
    25.0,
    float(STATE.brake_threshold),
    0.5,
    key="brake_threshold_slider",
)

STATE.ema_alpha = st.sidebar.slider(
    "EMA smoothing alpha",
    0.05,
    0.90,
    float(STATE.ema_alpha),
    0.05,
    key="ema_alpha_slider",
    help="Higher alpha responds faster but is less smooth.",
)

STATE.min_tap_hz = st.sidebar.slider(
    "Minimum tap frequency (Hz)",
    1.0,
    8.0,
    float(STATE.min_tap_hz),
    0.5,
    key="min_tap_hz_slider",
)

STATE.max_tap_hz = st.sidebar.slider(
    "Maximum tap frequency (Hz)",
    5.0,
    20.0,
    float(STATE.max_tap_hz),
    0.5,
    key="max_tap_hz_slider",
)

st.sidebar.divider()
st.sidebar.subheader("Calibration")

st.sidebar.caption(
    "At startup, hold both hands level in the yellow calibration box and sit neutrally for 3 seconds."
)

if st.sidebar.button("Start 3-second calibration", key="start_calibration_button"):
    with STATE.lock:
        STATE.calibrating = True
        STATE.calibrated = False
        STATE.calib_start = time.time()
        STATE.calib_hand_samples = []
        STATE.calib_torso_samples = []
    st.sidebar.success("Calibration started. Hold both hands visible and keep torso neutral.")

st.sidebar.markdown(
    f"""
    **Neutral baselines**
    - Wheel angle: `{STATE.neutral_angle:.1f}°`
    - Wrist distance: `{STATE.neutral_wrist_dist:.0f}px`
    - Torso lean: `{STATE.neutral_torso:.1f}°`
    """
)

st.sidebar.divider()
st.sidebar.subheader("Record live demo")

record_checkbox = st.sidebar.checkbox(
    "Record session to MP4 + CSV",
    value=False,
    key="record_session_checkbox",
    help="Records the annotated webcam frames to MP4 and logs gesture/key events to CSV.",
)

if record_checkbox:
    with STATE.lock:
        if not STATE.record:
            STATE.record = True
            STATE.log_event("recording", "start")
else:
    with STATE.lock:
        if STATE.record:
            STATE.record = False
            if STATE.writer is not None:
                try:
                    STATE.writer.release()
                except Exception:
                    pass
                STATE.writer = None
                STATE.record_size = None
            STATE.log_event("recording", "stop")

if not STATE.record and os.path.exists(STATE.video_path) and os.path.getsize(STATE.video_path) > 0:
    try:
        with open(STATE.video_path, "rb") as f:
            video_bytes = f.read()
        st.sidebar.download_button(
            "Download annotated demo MP4",
            data=video_bytes,
            file_name="gesture_kart_live_demo.mp4",
            mime="video/mp4",
            key="download_mp4_button",
        )
    except Exception as e:
        st.sidebar.warning(f"MP4 not ready yet: {e}")

if os.path.exists(STATE.csv_path):
    try:
        with open(STATE.csv_path, "rb") as f:
            csv_bytes = f.read()
        st.sidebar.download_button(
            "Download gesture-event CSV",
            data=csv_bytes,
            file_name="gesture_kart_event_log.csv",
            mime="text/csv",
            key="download_csv_button",
        )
    except Exception as e:
        st.sidebar.warning(f"CSV not ready yet: {e}")

st.sidebar.caption(f"MP4 path: `{STATE.video_path}`")
st.sidebar.caption(f"CSV path: `{STATE.csv_path}`")


# ----------------------------------------------------------------------------
# Main UI
# ----------------------------------------------------------------------------

st.title("🏎️ Gesture Kart: Two-Hand Wheel + Torso Lean Control")

st.caption(
    "Live MediaPipe Hands + Pose pipeline. The wrist line becomes a virtual steering wheel. "
    "Forward/backward torso lean becomes accelerate/brake. Browser JavaScript KeyboardEvents drive the embedded game."
)

with st.expander("2-minute live demo script", expanded=False):
    st.markdown(
        """
        **0:00 - 0:15 — Startup & calibration**  
        Press START on the webcam panel. Allow camera access. Hold both hands inside the yellow calibration box and sit neutral.

        **0:15 - 0:45 — Deliberate gesture confirmation**  
        Perform 4-5 slow two-hand steering rotations. Lean forward and backward. Confirm the badge, steering gauge, torso bar, and kart response.

        **0:45 - 1:30 — Natural gameplay speed**  
        Chain steering and lean gestures quickly. Watch the latency counter and tap/hold behavior.

        **1:30 - 2:00 — Edge cases**  
        Do small ambiguous wrist wobbles inside the deadzone. The label should show `Ignored: ambiguous wobble` and the kart should not misfire.
        """
    )

left_col, right_col = st.columns([11, 10], gap="large")

with left_col:
    st.subheader("Live webcam + gesture overlays")

    webrtc_streamer(
        key="gesture-kart-webrtc",
        mode=WEBRTC_MODE,
        video_processor_factory=GestureProcessor,
        media_stream_constraints=MEDIA_CONSTRAINTS,
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
            ]
        },
        async_processing=True,
    )

    st.caption(
        "Press **START** above, then allow camera access. For best results, use Chrome/Edge, "
        "show both hands clearly, and make sure shoulders/hips are visible for torso lean."
    )

with right_col:
    st.subheader("Embedded racing game + live telemetry")

    if SERVER_PORT is not None:
        game_html = GAME_HTML_TEMPLATE.replace("__PORT__", str(SERVER_PORT))
        components.html(game_html, height=730, scrolling=False)
    else:
        st.error(
            "Could not start the local JSON bridge for browser key dispatch. "
            "Close other apps using ports 8765-8794 and restart."
        )

st.divider()

st.markdown(
    """
    ### How it works
    1. `streamlit-webrtc` captures live webcam frames.
    2. MediaPipe Hands detects both wrists and computes the virtual wheel angle.
    3. EMA smoothing removes jitter and maps angle to a continuous steering value.
    4. MediaPipe Pose computes torso lean relative to your calibrated neutral posture.
    5. The app emits browser `KeyboardEvent` arrow-key taps/holds to the embedded game.
    6. The sidebar switches between **digital tap steering** and **proportional key-hold steering**.
    """
)