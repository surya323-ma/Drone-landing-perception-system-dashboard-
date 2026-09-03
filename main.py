#Dev/Creator=tubakhxn

import os
import sys
import subprocess
import importlib


def _ensure(pkg_import, pip_name=None):
    pip_name = pip_name or pkg_import
    try:
        importlib.import_module(pkg_import)
        return True
    except ImportError:
        print(f"[setup] Installing missing dependency: {pip_name} ...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet",
                 "--break-system-packages", pip_name]
            )
            importlib.import_module(pkg_import)
            return True
        except Exception as e:
            print(f"[setup] WARNING: could not install {pip_name}: {e}")
            return False


_ensure("numpy")
_ensure("cv2", "opencv-contrib-python")
_ensure("scipy")


_YOLO_AVAILABLE = _ensure("ultralytics")

import math
import time
import glob
import random
from collections import deque

import numpy as np
import cv2

try:
    from scipy.optimize import linear_sum_assignment
    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False



class Palette:
    CYAN        = (255, 240, 60)      # BGR
    CYAN_SOFT   = (230, 200, 90)
    TEAL        = (200, 180, 40)
    GREEN       = (140, 235, 120)
    GREEN_DIM   = (90, 160, 80)
    WHITE       = (245, 245, 245)
    WHITE_DIM   = (190, 190, 190)
    RED_WARN    = (80, 80, 235)
    AMBER       = (60, 170, 235)
    BG_PANEL    = (28, 22, 18)
    GRID        = (90, 70, 50)


FRAME_MARGIN = 24


def clamp(v, a, b):
    return max(a, min(b, v))


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_pt(p0, p1, t):
    return (lerp(p0[0], p1[0], t), lerp(p0[1], p1[1], t))


def ease(t):
    t = clamp(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)

class OneEuroFilter:
    def __init__(self, freq=30.0, min_cutoff=1.2, beta=0.02, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, freq):
        te = 1.0 / freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x, t=None):
        if t is None:
            t = time.time()
        if self.t_prev is None:
            self.t_prev = t
        dt = max(t - self.t_prev, 1e-3)
        self.freq = 1.0 / dt if dt > 0 else self.freq
        self.t_prev = t

        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            return x

        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff, self.freq)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, self.freq)
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


class Vec2EuroFilter:
    """One-Euro filter applied independently to x and y."""
    def __init__(self, min_cutoff=1.2, beta=0.02):
        self.fx = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self.fy = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)

    def __call__(self, pt, t=None):
        return (self.fx(pt[0], t), self.fy(pt[1], t))


class KalmanBox2D:
    """Constant-velocity Kalman filter over (cx, cy, w, h)."""
    def __init__(self):
        self.kf = cv2.KalmanFilter(8, 4)
        dt = 1.0
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, dt, 0, 0, 0],
            [0, 1, 0, 0, 0, dt, 0, 0],
            [0, 0, 1, 0, 0, 0, dt, 0],
            [0, 0, 0, 1, 0, 0, 0, dt],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)
        self.initialized = False

    def init(self, cx, cy, w, h):
        self.kf.statePost = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)
        self.initialized = True

    def predict(self):
        s = self.kf.predict()
        s = s.flatten()
        return float(s[0]), float(s[1]), float(s[2]), float(s[3])

    def correct(self, cx, cy, w, h):
        meas = np.array([cx, cy, w, h], dtype=np.float32)
        self.kf.correct(meas)



class YoloDroneDetector:
   
    ACCEPT_CLASSES = {"airplane", "bird", "kite", "frisbee"}

    def __init__(self):
        self.ok = False
        self.model = None
        if not _YOLO_AVAILABLE:
            return
        try:
            from ultralytics import YOLO
            self.model = YOLO("yolov8n.pt")
            self.ok = True
        except Exception as e:
            print(f"[detector] YOLO unavailable, using fallback only: {e}")
            self.ok = False

    def detect(self, frame_bgr, conf=0.08):
        if not self.ok:
            return []
        try:
            res = self.model.predict(frame_bgr, verbose=False, conf=conf, imgsz=640)
        except Exception:
            return []
        out = []
        for r in res:
            for b in r.boxes:
                cls_id = int(b.cls[0])
                name = self.model.names.get(cls_id, "")
                if name not in self.ACCEPT_CLASSES:
                    continue
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                out.append((x1, y1, x2, y2, float(b.conf[0])))
        return out


class MotionContrastDetector:
    

    def __init__(self, frame_shape):
        h, w = frame_shape[:2]
        self.h, self.w = h, w
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=120, varThreshold=18, detectShadows=False)
        self.prev_gray = None
        self.min_area = max(30, int(w * h * 0.00004))
        self.max_area = int(w * h * 0.06)

    def detect(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        fg = self.bg_sub.apply(gray_blur, learningRate=0.01)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        flow_mask = np.zeros_like(gray)
        if self.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray_blur, None, 0.5, 2, 15, 2, 5, 1.1, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mag_n = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            _, flow_mask = cv2.threshold(mag_n.astype(np.uint8), 40, 255, cv2.THRESH_BINARY)
        self.prev_gray = gray_blur

    
        edges = cv2.Canny(gray_blur, 60, 160)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        combined = cv2.bitwise_or(fg, flow_mask)
        combined = cv2.morphologyEx(
            combined, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
        combined = cv2.morphologyEx(
            combined, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    
        roi_mask = np.zeros_like(combined)
        roi_mask[:int(h * 0.85), :] = 255
        combined = cv2.bitwise_and(combined, roi_mask)

        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area or area > self.max_area:
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            if ch == 0:
                continue
            aspect = cw / float(ch)
            if aspect < 0.25 or aspect > 4.5:
                continue
     
            roi_edges = edges[y:y + ch, x:x + cw]
            edge_density = float(np.count_nonzero(roi_edges)) / max(1, cw * ch)
            score = area * (0.4 + edge_density * 3.0)
            candidates.append((x, y, x + cw, y + ch, score))

        candidates.sort(key=lambda t: t[4], reverse=True)
        return candidates[:5]


class DroneTracker:


    STATES = ["SEARCHING", "ACQUIRED", "TRACKING", "LOCKED", "APPROACHING"]

    def __init__(self, frame_shape, use_yolo=True):
        self.h, self.w = frame_shape[:2]
        self.yolo = YoloDroneDetector() if use_yolo else None
        self.motion = MotionContrastDetector(frame_shape)

        self.kf = KalmanBox2D()
        self.center_filter = Vec2EuroFilter(min_cutoff=1.0, beta=0.015)
        self.size_filter = OneEuroFilter(min_cutoff=1.0, beta=0.01)

        self.track_id = random.randint(1000, 9999)
        self.state = "SEARCHING"
        self.frames_since_hit = 999
        self.hits = 0
        self.confidence = 0.0         
        self.age = 0

        self.last_box = None          
        self.last_center = None
        self.last_size = None
        self.trail = deque(maxlen=45)  
        self.scale_history = deque(maxlen=60)

    def _pick_best(self, yolo_boxes, motion_boxes):
    
        if not yolo_boxes and not motion_boxes:
            return None, 0.0

        def to_xywh(b):
            x1, y1, x2, y2 = b[:4]
            return x1, y1, x2, y2

        if yolo_boxes and motion_boxes:
            yb = max(yolo_boxes, key=lambda b: b[4])
            mb = motion_boxes[0]
            yx1, yy1, yx2, yy2, _ = yb
            mx1, my1, mx2, my2, _ = mb
            iou = self._iou((yx1, yy1, yx2, yy2), (mx1, my1, mx2, my2))
            if iou > 0.1:
                bx = ((yx1 + mx1) / 2, (yy1 + my1) / 2,
                      (yx2 + mx2) / 2, (yy2 + my2) / 2)
                return bx, 0.9
            else:
                return to_xywh(mb), 0.65

        if motion_boxes:
            return to_xywh(motion_boxes[0]), 0.55
        yb = max(yolo_boxes, key=lambda b: b[4])
        return to_xywh(yb), 0.5

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def update(self, frame_bgr, t_sec):
        self.age += 1
        yolo_boxes = self.yolo.detect(frame_bgr) if (self.yolo and self.yolo.ok) else []
        motion_boxes = self.motion.detect(frame_bgr)

        raw_box, det_conf = self._pick_best(yolo_boxes, motion_boxes)

        if raw_box is not None:
            x1, y1, x2, y2 = raw_box
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            bw, bh = max(4.0, x2 - x1), max(4.0, y2 - y1)

            if not self.kf.initialized:
                self.kf.init(cx, cy, bw, bh)
            else:
                self.kf.predict()
                self.kf.correct(cx, cy, bw, bh)

            self.frames_since_hit = 0
            self.hits += 1
            self.confidence = lerp(self.confidence, det_conf, 0.25)
        else:
            if self.kf.initialized:
                self.kf.predict()
            self.frames_since_hit += 1
            self.confidence = lerp(self.confidence, 0.0, 0.12)

        if self.kf.initialized:
            kcx, kcy, kw, kh = (self.kf.kf.statePost[0], self.kf.kf.statePost[1],
                                 self.kf.kf.statePost[2], self.kf.kf.statePost[3])
            kw = max(6.0, float(kw)); kh = max(6.0, float(kh))
            kcx, kcy = float(kcx), float(kcy)

            s_cx, s_cy = self.center_filter((kcx, kcy), t=t_sec)
            s_size = self.size_filter(math.sqrt(kw * kh), t=t_sec)
            aspect = kw / kh if kh > 0 else 1.0
            s_w = s_size * math.sqrt(aspect)
            s_h = s_size / math.sqrt(aspect)

            s_cx = clamp(s_cx, 0, self.w)
            s_cy = clamp(s_cy, 0, self.h)

            jump_ok = True
            if self.trail:
                px, py = self.trail[-1]
                jump_dist = math.hypot(s_cx - px, s_cy - py)
                diag = math.hypot(self.w, self.h)
                if jump_dist > diag * 0.045:
                    jump_ok = False

            self.last_center = (s_cx, s_cy)
            self.last_size = (s_w, s_h)
            self.last_box = (s_cx - s_w / 2, s_cy - s_h / 2,
                              s_cx + s_w / 2, s_cy + s_h / 2)
            if not jump_ok:
                self.trail.clear()
            min_step = math.hypot(self.w, self.h) * 0.004
            if not self.trail or math.hypot(
                    s_cx - self.trail[-1][0], s_cy - self.trail[-1][1]) > min_step:
                self.trail.append(self.last_center)
            self.scale_history.append(s_w * s_h)

        self._update_state()
        return self.last_box, self.confidence, self.state

    def _update_state(self):
        target = self.state
        if self.confidence < 0.12 and self.frames_since_hit > 15:
            target = "SEARCHING"
        elif self.confidence < 0.35:
            target = "ACQUIRED"
        elif self.confidence < 0.6:
            target = "TRACKING"
        elif self.confidence < 0.8:
            target = "LOCKED"
        else:
            target = "APPROACHING"

        order = self.STATES
        cur_i = order.index(self.state)
        tgt_i = order.index(target)
        if tgt_i > cur_i:
            self.state = order[min(cur_i + 1, tgt_i)]
        elif tgt_i < cur_i and self.frames_since_hit > 20:
            self.state = order[max(cur_i - 1, tgt_i)]

    @property
    def is_visible(self):
        return self.last_box is not None and self.frames_since_hit < 45

    def scale_trend(self):
        """>0 growing (approaching), <0 shrinking (receding)."""
        if len(self.scale_history) < 10:
            return 0.0
        a = np.mean(list(self.scale_history)[:len(self.scale_history)//2])
        b = np.mean(list(self.scale_history)[len(self.scale_history)//2:])
        if a <= 1e-3:
            return 0.0
        return clamp((b - a) / a, -1.0, 1.0)



class TelemetryEstimator:
    def __init__(self, frame_shape):
        self.h, self.w = frame_shape[:2]
        self.f_distance = OneEuroFilter(min_cutoff=0.8, beta=0.01)
        self.f_altitude = OneEuroFilter(min_cutoff=0.8, beta=0.01)
        self.f_angle = OneEuroFilter(min_cutoff=0.8, beta=0.01)
        self.f_align = OneEuroFilter(min_cutoff=0.8, beta=0.01)
        self.f_lat = OneEuroFilter(min_cutoff=0.8, beta=0.01)
        self.f_vert = OneEuroFilter(min_cutoff=0.8, beta=0.01)
        self.f_conf = OneEuroFilter(min_cutoff=0.6, beta=0.01)

    def estimate(self, center, size, track_conf, scale_trend, t_sec):
        if center is None or size is None:
            return None

        cx, cy = center
        sw, sh = size
        diag = math.sqrt(sw * sw + sh * sh)
        ref_diag = 0.05 * math.sqrt(self.w ** 2 + self.h ** 2)

        raw_distance = clamp(ref_diag / max(diag, 1e-3) * 8.0, 3.0, 120.0)
        distance = self.f_distance(raw_distance, t=t_sec)

        norm_y = 1.0 - clamp(cy / self.h, 0.0, 1.0)
        raw_altitude = 2.0 + norm_y * 40.0 + scale_trend * 3.0
        altitude = self.f_altitude(raw_altitude, t=t_sec)

        norm_x_off = (cx - self.w / 2.0) / (self.w / 2.0)
        raw_angle = clamp(norm_x_off * 12.0, -25.0, 25.0)
        angle = self.f_angle(raw_angle, t=t_sec)

        raw_align = clamp(100.0 - abs(norm_x_off) * 60.0 - abs(angle) * 0.8, 30.0, 99.0)
        align = self.f_align(raw_align, t=t_sec)

        raw_lat = norm_x_off * 3.2
        lateral = self.f_lat(raw_lat, t=t_sec)

        norm_y_off = (cy - self.h * 0.4) / (self.h * 0.5)
        raw_vert = clamp(norm_y_off * 2.5, -4.0, 4.0)
        vertical = self.f_vert(raw_vert, t=t_sec)

        raw_conf = clamp(track_conf * 100.0 * (0.6 + 0.4 * (1 - abs(norm_x_off))), 0, 99)
        landing_conf = self.f_conf(raw_conf, t=t_sec)

        return {
            "distance": distance,
            "altitude": altitude,
            "angle": angle,
            "alignment": align,
            "lateral": lateral,
            "vertical": vertical,
            "landing_confidence": landing_conf,
        }



class ApproachCorridor:
    """Projects a trapezoidal 3D-style corridor from the drone position
    down to a landing zone quadrilateral, with smoothing on every corner
    so geometry glides rather than snapping."""

    def __init__(self, frame_shape):
        self.h, self.w = frame_shape[:2]
        self.filters = {k: Vec2EuroFilter(min_cutoff=0.7, beta=0.02)
                         for k in ["apex", "tl", "tr", "bl", "br"]}
        self.landing_center_filter = Vec2EuroFilter(min_cutoff=0.6, beta=0.02)
        self.t_accum = 0.0

    def update(self, drone_center, drone_size, t_sec):
        self.t_accum = t_sec
        if drone_center is None:
            drone_center = (self.w * 0.5, self.h * 0.28)
            drone_size = (self.w * 0.05, self.w * 0.05)

        cx, cy = drone_center
        sw, sh = drone_size
        apex = (cx, cy + sh * 0.55)


        closeness = clamp((sw * sh) / (0.02 * self.w * self.h), 0.0, 1.0)
        zone_y = lerp(self.h * 0.72, self.h * 0.90, closeness)
        zone_half_w = lerp(self.w * 0.16, self.w * 0.30, closeness)
        zone_depth = lerp(self.h * 0.10, self.h * 0.16, closeness)

        drift = (cx - self.w * 0.5) * 0.35
        zone_cx = clamp(self.w * 0.5 + drift, zone_half_w + 20, self.w - zone_half_w - 20)

        near_half = zone_half_w
        far_half = zone_half_w * 0.62
        tl = (zone_cx - far_half, zone_y - zone_depth * 0.5)
        tr = (zone_cx + far_half, zone_y - zone_depth * 0.5)
        bl = (zone_cx - near_half, zone_y + zone_depth * 0.5)
        br = (zone_cx + near_half, zone_y + zone_depth * 0.5)

        apex_s = self.filters["apex"](apex, t=t_sec)
        tl_s = self.filters["tl"](tl, t=t_sec)
        tr_s = self.filters["tr"](tr, t=t_sec)
        bl_s = self.filters["bl"](bl, t=t_sec)
        br_s = self.filters["br"](br, t=t_sec)
        center_s = self.landing_center_filter((zone_cx, zone_y), t=t_sec)

        return {
            "apex": apex_s, "tl": tl_s, "tr": tr_s, "bl": bl_s, "br": br_s,
            "center": center_s, "closeness": closeness,
        }



def draw_dashed_line(img, p1, p2, color, thickness=2, dash_len=10, gap_len=8,
                      phase=0.0):
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    dist = np.linalg.norm(p2 - p1)
    if dist < 1:
        return
    direction = (p2 - p1) / dist
    step = dash_len + gap_len
    start = -(phase % step)
    d = start
    while d < dist:
        seg_start = max(d, 0)
        seg_end = min(d + dash_len, dist)
        if seg_end > seg_start:
            a = p1 + direction * seg_start
            b = p1 + direction * seg_end
            cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)),
                      color, thickness, cv2.LINE_AA)
        d += step


def draw_glow_circle(img, center, radius, color, intensity=1.0):
    overlay = img.copy()
    for r, a in [(radius * 2.2, 0.10), (radius * 1.5, 0.18), (radius, 0.35)]:
        cv2.circle(overlay, (int(center[0]), int(center[1])), int(r), color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.5 * intensity, img, 1 - 0.5 * intensity, 0, img)
    cv2.circle(img, (int(center[0]), int(center[1])), max(2, int(radius * 0.55)),
               color, -1, cv2.LINE_AA)


def draw_pin_marker(img, pos, color, scale=1.0, pulse=0.0):
    x, y = int(pos[0]), int(pos[1])
    r = int(9 * scale)
    stem = int(16 * scale)
    pts = np.array([
        [x, y + stem],
        [x - r, y + stem - r],
        [x - r, y + stem - 2 * r],
        [x, y + stem - int(2.6 * r)],
        [x + r, y + stem - 2 * r],
        [x + r, y + stem - r],
    ], dtype=np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], color, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    cv2.polylines(img, [pts], True, Palette.WHITE, 1, cv2.LINE_AA)
    pulse_r = int(r * (1.3 + pulse * 1.4))
    cv2.circle(img, (x, y + stem - 2 * r), pulse_r, color, 1, cv2.LINE_AA)
    cv2.circle(img, (x, y + stem - 2 * r), max(2, int(r * 0.4)), Palette.WHITE, -1, cv2.LINE_AA)


def rounded_rect(img, pt1, pt2, color, thickness=-1, radius=10, alpha=1.0):
    x1, y1 = pt1
    x2, y2 = pt2
    overlay = img.copy()
    if thickness < 0:
        cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1, cv2.LINE_AA)
        cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1, cv2.LINE_AA)
        for cx, cy in [(x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                        (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)]:
            cv2.circle(overlay, (cx, cy), radius, color, -1, cv2.LINE_AA)
    else:
        cv2.rectangle(overlay, pt1, pt2, color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def put_text(img, text, org, scale=0.5, color=Palette.WHITE, thickness=1,
             font=cv2.FONT_HERSHEY_SIMPLEX, glow=False):
    if glow:
        cv2.putText(img, text, org, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)



class HUDRenderer:
    def __init__(self, frame_shape, fps_target=30):
        self.h, self.w = frame_shape[:2]
        self.fps_target = fps_target
        self.t0 = time.time()
        self.frame_count = 0
        self._fps_smooth = fps_target
        self._radar_history = deque(maxlen=60)
        self.corridor_alpha_f = OneEuroFilter(min_cutoff=0.5, beta=0.02)
        self.zone_scan_phase = 0.0

    def render(self, frame, track, telemetry, corridor, t_sec, real_fps=None):
        h, w = self.h, self.w
        out = frame.copy()
        overlay_full = np.zeros_like(out)

        visible = track.is_visible
        conf = track.confidence
        pulse = 0.5 + 0.5 * math.sin(t_sec * 3.2)

        self._draw_corridor(overlay_full, corridor, visible, conf, t_sec)

        alpha_target = 0.85 if visible else 0.35
        alpha = self.corridor_alpha_f(alpha_target, t=t_sec)
        cv2.addWeighted(overlay_full, alpha, out, 1.0, 0, out)

        if visible:
            self._draw_drone_box(out, track, t_sec, pulse)

        self._draw_trail(out, track)

        self._draw_scanlines(out, t_sec)

        self._draw_vignette_frame(out)
        self._draw_top_left(out, track, real_fps)
        self._draw_top_right(out, track, telemetry)
        self._draw_bottom_left(out, telemetry)
        self._draw_radar_panel(out, track, corridor, t_sec)
        self._draw_center_reticle(out, track, t_sec)

        self.frame_count += 1
        return out

    def _draw_corridor(self, img, corridor, visible, conf, t_sec):
        apex, tl, tr, bl, br = (corridor["apex"], corridor["tl"], corridor["tr"],
                                 corridor["bl"], corridor["br"])
        closeness = corridor["closeness"]

        base_alpha = 0.16 + 0.10 * closeness
        color = Palette.CYAN if visible else Palette.WHITE_DIM

        poly_left = np.array([apex, tl, bl], dtype=np.int32)
        poly_right = np.array([apex, tr, br], dtype=np.int32)
        poly_far = np.array([tl, tr, br, bl], dtype=np.int32)

        fill = img.copy()
        cv2.fillPoly(fill, [poly_left], color, cv2.LINE_AA)
        cv2.fillPoly(fill, [poly_right], color, cv2.LINE_AA)
        cv2.fillPoly(fill, [poly_far], Palette.GREEN, cv2.LINE_AA)
        cv2.addWeighted(fill, base_alpha, img, 1 - base_alpha, 0, img)

        phase = (t_sec * 40) % 200
        draw_dashed_line(img, apex, tl, color, 2, 12, 8, phase)
        draw_dashed_line(img, apex, tr, color, 2, 12, 8, phase)
        draw_dashed_line(img, apex, bl, color, 2, 12, 8, phase * 0.8)
        draw_dashed_line(img, apex, br, color, 2, 12, 8, phase * 0.8)

        mid_far = lerp_pt(tl, tr, 0.5)
        mid_near = lerp_pt(bl, br, 0.5)
        cv2.line(img, (int(apex[0]), int(apex[1])),
                  (int(mid_near[0]), int(mid_near[1])), Palette.WHITE, 1, cv2.LINE_AA)
        draw_dashed_line(img, mid_far, mid_near, Palette.CYAN_SOFT, 1, 6, 6, phase)

        for f in (0.33, 0.66):
            l = lerp_pt(apex, bl, f)
            r = lerp_pt(apex, br, f)
            draw_dashed_line(img, l, r, Palette.TEAL, 1, 8, 6, phase)

        self.zone_scan_phase = (self.zone_scan_phase + 2.4) % 300
        draw_dashed_line(img, tl, tr, Palette.GREEN, 2, 10, 6, self.zone_scan_phase)
        draw_dashed_line(img, tr, br, Palette.GREEN, 2, 10, 6, self.zone_scan_phase)
        draw_dashed_line(img, br, bl, Palette.GREEN, 2, 10, 6, self.zone_scan_phase)
        draw_dashed_line(img, bl, tl, Palette.GREEN, 2, 10, 6, self.zone_scan_phase)

        for f in (0.33, 0.66):
            a = lerp_pt(tl, bl, f)
            b = lerp_pt(tr, br, f)
            cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                      Palette.GREEN_DIM, 1, cv2.LINE_AA)
        for f in (0.33, 0.66):
            a = lerp_pt(tl, tr, f)
            b = lerp_pt(bl, br, f)
            cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                      Palette.GREEN_DIM, 1, cv2.LINE_AA)

        sweep_t = (math.sin(t_sec * 1.1) + 1) / 2
        sweep_l = lerp_pt(tl, bl, sweep_t)
        sweep_r = lerp_pt(tr, br, sweep_t)
        cv2.line(img, (int(sweep_l[0]), int(sweep_l[1])),
                  (int(sweep_r[0]), int(sweep_r[1])), Palette.GREEN, 2, cv2.LINE_AA)

        pulse = 0.5 + 0.5 * math.sin(t_sec * 3.0)
        for corner in (tl, tr, bl, br):
            draw_pin_marker(img, corner, Palette.GREEN, scale=0.85, pulse=pulse)

        center = corridor["center"]
        label = "LANDING ZONE"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        put_text(img, label, (int(center[0] - tw / 2), int(center[1] + th / 2)),
                 0.45, Palette.WHITE, 1, glow=True)

    def _draw_drone_box(self, img, track, t_sec, pulse):
        x1, y1, x2, y2 = [int(v) for v in track.last_box]
        color = Palette.CYAN
        L = max(10, int((x2 - x1) * 0.18))
        thick = 2

        for (px, py, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                                   (x1, y2, 1, -1), (x2, y2, -1, -1)]:
            cv2.line(img, (px, py), (px + dx * L, py), color, thick, cv2.LINE_AA)
            cv2.line(img, (px, py), (px, py + dy * L), color, thick, cv2.LINE_AA)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

        cx, cy = track.last_center
        cv2.circle(img, (int(cx), int(cy)), 2, Palette.WHITE, -1, cv2.LINE_AA)
        ring_r = int(6 + 3 * pulse)
        cv2.circle(img, (int(cx), int(cy)), ring_r, color, 1, cv2.LINE_AA)

        label = f"DRONE #{track.track_id}  {track.state}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = max(18, y1 - 10)
        rounded_rect(img, (x1 - 4, ly - th - 8), (x1 + tw + 8, ly + 4),
                     Palette.BG_PANEL, -1, 6, 0.55)
        put_text(img, label, (x1, ly), 0.5, color, 1)

    def _draw_trail(self, img, track):
        pts = list(track.trail)
        n = len(pts)
        if n < 2:
            return
        for i in range(1, n):
            a = pts[i - 1]
            b = pts[i]
            t = i / n
            alpha = 0.15 + 0.55 * t
            color = tuple(int(c * alpha + 0 * (1 - alpha)) for c in Palette.CYAN_SOFT)
            cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                      color, 1, cv2.LINE_AA)

    def _draw_scanlines(self, img, t_sec):
        h, w = self.h, self.w
        y = int((math.sin(t_sec * 0.35) * 0.5 + 0.5) * h)
        overlay = img.copy()
        cv2.line(overlay, (0, y), (w, y), Palette.CYAN, 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.08, img, 0.92, 0, img)

    def _draw_vignette_frame(self, img):
        h, w = self.h, self.w
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 78), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

        m = FRAME_MARGIN
        corner_len = 26
        color = Palette.CYAN_SOFT
        for (x, y, dx, dy) in [(m, m, 1, 1), (w - m, m, -1, 1),
                                 (m, h - m, 1, -1), (w - m, h - m, -1, -1)]:
            cv2.line(img, (x, y), (x + dx * corner_len, y), color, 2, cv2.LINE_AA)
            cv2.line(img, (x, y), (x, y + dy * corner_len), color, 2, cv2.LINE_AA)

    def _draw_top_left(self, img, track, real_fps):
        x, y = FRAME_MARGIN + 14, 30
        put_text(img, "DRONE LANDING PERCEPTION", (x, y), 0.62, Palette.WHITE, 2, glow=True)
        y += 22
        fps = real_fps if real_fps else self.fps_target
        self._fps_smooth = lerp(self._fps_smooth, fps, 0.1)
        put_text(img, f"FPS {self._fps_smooth:4.1f}   FRAME {self.frame_count:05d}",
                 (x, y), 0.42, Palette.WHITE_DIM, 1, glow=True)
        y += 18
        state_color = {
            "SEARCHING": Palette.AMBER, "ACQUIRED": Palette.AMBER,
            "TRACKING": Palette.CYAN, "LOCKED": Palette.GREEN,
            "APPROACHING": Palette.GREEN,
        }.get(track.state, Palette.WHITE)
        put_text(img, f"TRACKING STATUS: {track.state}", (x, y), 0.46, state_color, 1, glow=True)
        y += 18
        put_text(img, f"TRACK ID: {track.track_id}", (x, y), 0.42, Palette.WHITE_DIM, 1, glow=True)

    def _draw_top_right(self, img, track, telemetry):
        w = self.w
        x = w - FRAME_MARGIN - 14
        y = 30
        lock_txt = "TARGET LOCK: YES" if track.state in ("LOCKED", "APPROACHING") else "TARGET LOCK: NO"
        lock_color = Palette.GREEN if track.state in ("LOCKED", "APPROACHING") else Palette.AMBER
        (tw, _), _ = cv2.getTextSize(lock_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        put_text(img, lock_txt, (x - tw, y), 0.5, lock_color, 1, glow=True)
        y += 22

        conf_pct = track.confidence * 100
        bar_label = f"CONFIDENCE {conf_pct:5.1f}%"
        (tw, _), _ = cv2.getTextSize(bar_label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        put_text(img, bar_label, (x - tw, y), 0.42, Palette.WHITE_DIM, 1, glow=True)
        bar_w, bar_h = 120, 6
        bx2 = x
        bx1 = bx2 - bar_w
        by1 = y + 6
        cv2.rectangle(img, (bx1, by1), (bx2, by1 + bar_h), Palette.GRID, 1, cv2.LINE_AA)
        fill_w = int(bar_w * clamp(track.confidence, 0, 1))
        if fill_w > 0:
            cv2.rectangle(img, (bx1, by1), (bx1 + fill_w, by1 + bar_h),
                          Palette.CYAN, -1, cv2.LINE_AA)
        y += 22

        align_val = telemetry["alignment"] if telemetry else 0.0
        align_txt = f"ALIGNMENT {align_val:5.1f}%"
        (tw, _), _ = cv2.getTextSize(align_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        put_text(img, align_txt, (x - tw, y), 0.42, Palette.WHITE_DIM, 1, glow=True)
        y += 20

        landing_state = track.state if track.state != "SEARCHING" else "IDLE"
        ls_txt = f"LANDING STATE: {landing_state}"
        (tw, _), _ = cv2.getTextSize(ls_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        put_text(img, ls_txt, (x - tw, y), 0.42, Palette.WHITE_DIM, 1, glow=True)

    def _draw_bottom_left(self, img, telemetry):
        x = FRAME_MARGIN + 14
        y = self.h - 128
        panel_w = 250
        rounded_rect(img, (x - 10, y - 20), (x + panel_w, self.h - 16),
                     Palette.BG_PANEL, -1, 8, 0.45)
        put_text(img, "TELEMETRY (EST.)", (x, y), 0.42, Palette.CYAN_SOFT, 1)
        y += 18

        if telemetry is None:
            put_text(img, "-- NO TARGET --", (x, y), 0.42, Palette.WHITE_DIM, 1)
            return

        rows = [
            ("DISTANCE", f"{telemetry['distance']:5.1f} m"),
            ("ALTITUDE", f"{telemetry['altitude']:5.1f} m"),
            ("APPROACH ANGLE", f"{telemetry['angle']:+5.1f} deg"),
            ("ALIGNMENT", f"{telemetry['alignment']:5.1f}%"),
            ("LATERAL OFFSET", f"{telemetry['lateral']:+4.2f} m"),
            ("VERTICAL OFFSET", f"{telemetry['vertical']:+4.2f} m"),
            ("LANDING CONFIDENCE", f"{telemetry['landing_confidence']:5.1f}%"),
        ]
        for label, val in rows:
            put_text(img, label, (x, y), 0.37, Palette.WHITE_DIM, 1)
            (tw, _), _ = cv2.getTextSize(val, cv2.FONT_HERSHEY_SIMPLEX, 0.37, 1)
            put_text(img, val, (x + panel_w - 20 - tw, y), 0.37, Palette.WHITE, 1)
            y += 15

    def _draw_radar_panel(self, img, track, corridor, t_sec):
        w, h = self.w, self.h
        panel_size = 150
        px2, py2 = w - FRAME_MARGIN - 14, h - 16
        px1, py1 = px2 - panel_size, py2 - panel_size

        rounded_rect(img, (px1, py1), (px2, py2), Palette.BG_PANEL, -1, 8, 0.5)
        cv2.rectangle(img, (px1, py1), (px2, py2), Palette.GRID, 1, cv2.LINE_AA)
        put_text(img, "APPROACH RADAR", (px1 + 8, py1 + 16), 0.35, Palette.CYAN_SOFT, 1)

        cx, cy = px1 + panel_size // 2, py1 + panel_size // 2 + 12
        max_r = panel_size // 2 - 24
        for rr in (max_r, int(max_r * 0.66), int(max_r * 0.33)):
            cv2.circle(img, (cx, cy), rr, Palette.GRID, 1, cv2.LINE_AA)
        cv2.line(img, (cx - max_r, cy), (cx + max_r, cy), Palette.GRID, 1, cv2.LINE_AA)
        cv2.line(img, (cx, cy - max_r), (cx, cy + max_r), Palette.GRID, 1, cv2.LINE_AA)

        sweep_ang = (t_sec * 90) % 360
        ex = cx + int(max_r * math.cos(math.radians(sweep_ang)))
        ey = cy + int(max_r * math.sin(math.radians(sweep_ang)))
        overlay = img.copy()
        cv2.line(overlay, (cx, cy), (ex, ey), Palette.GREEN, 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

        tx, ty = cx, cy + max_r - 6
        cv2.drawMarker(img, (tx, ty), Palette.GREEN, cv2.MARKER_TRIANGLE_UP, 8, 2)

        if track.is_visible:
            dc = track.last_center
            zc = corridor["center"]
            rel_x = (dc[0] - zc[0]) / (self.w * 0.35)
            rel_y = (dc[1] - zc[1]) / (self.h * 0.35)
            rx = clamp(cx + rel_x * max_r, px1 + 6, px2 - 6)
            ry = clamp(cy + rel_y * max_r, py1 + 20, py2 - 6)
            self._radar_history.append((rx, ry))
        pts = list(self._radar_history)
        for i in range(1, len(pts)):
            t = i / max(1, len(pts))
            color = tuple(int(c * (0.2 + 0.5 * t)) for c in Palette.CYAN)
            cv2.line(img, (int(pts[i-1][0]), int(pts[i-1][1])),
                      (int(pts[i][0]), int(pts[i][1])), color, 1, cv2.LINE_AA)
        if pts:
            pulse = 0.5 + 0.5 * math.sin(t_sec * 4)
            draw_glow_circle(img, pts[-1], 3 + pulse, Palette.CYAN, 0.8)

    def _draw_center_reticle(self, img, track, t_sec):
        pass


def open_writer(path, fps, size):
    fourcc_options = ["mp4v", "avc1", "H264"]
    for fcc in fourcc_options:
        fourcc = cv2.VideoWriter_fourcc(*fcc)
        vw = cv2.VideoWriter(path, fourcc, fps, size)
        if vw.isOpened():
            return vw
        vw.release()
    # last resort
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    return cv2.VideoWriter(path.replace(".mp4", ".avi"), fourcc, fps, size)


def ensure_dirs():
    os.makedirs("output", exist_ok=True)



def _telemetry_record(frame_idx, t_sec, tracker, telemetry, real_fps):
    """Build a single JSON-serializable telemetry sample for the dashboard API."""
    rec = {
        "frame": frame_idx,
        "t": round(t_sec, 3),
        "state": tracker.state,
        "track_id": tracker.track_id,
        "confidence": round(float(tracker.confidence) * 100.0, 1),
        "visible": bool(tracker.is_visible),
        "fps": round(float(real_fps), 1) if real_fps else None,
    }
    if telemetry:
        rec.update({
            "distance_m": round(telemetry["distance"], 1),
            "altitude_m": round(telemetry["altitude"], 1),
            "approach_angle_deg": round(telemetry["angle"], 1),
            "alignment_pct": round(telemetry["alignment"], 1),
            "lateral_offset_m": round(telemetry["lateral"], 2),
            "vertical_offset_m": round(telemetry["vertical"], 2),
            "landing_confidence_pct": round(telemetry["landing_confidence"], 1),
        })
    if tracker.last_center is not None:
        rec["center"] = [round(tracker.last_center[0], 1), round(tracker.last_center[1], 1)]
    return rec


def process_video(input_path, progress_cb=None, telemetry_cb=None):
    """Run the perception pipeline over a video file.

    progress_cb(frame_idx, total_frames, state, confidence) is called periodically
    while processing so a caller (e.g. the dashboard API) can report status.
    telemetry_cb(record_dict) is called once per processed frame with a compact,
    JSON-serializable telemetry sample so a caller can stream/store the full run.
    Both callbacks are optional and the CLI entry point (main()) does not pass them,
    so command-line behaviour is unchanged.
    """
    if not os.path.isfile(input_path):
        print(f"[error] Input video not found: {input_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[error] Could not open video: {input_path}")
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 1 or src_fps > 120:
        src_fps = 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if src_w <= 0 or src_h <= 0:
        print("[error] Invalid video dimensions.")
        sys.exit(1)

 
    proc_w, proc_h = src_w, src_h
    max_dim = 1280
    scale = 1.0
    if max(src_w, src_h) > max_dim:
        scale = max_dim / max(src_w, src_h)
        proc_w, proc_h = int(src_w * scale), int(src_h * scale)
        proc_w -= proc_w % 2
        proc_h -= proc_h % 2

    ensure_dirs()
    base = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join("output", f"{base}_perception.mp4")
    writer = open_writer(out_path, src_fps, (proc_w, proc_h))
    if not writer.isOpened():
        print("[error] Could not open VideoWriter with any codec.")
        sys.exit(1)

    tracker = DroneTracker((proc_h, proc_w), use_yolo=True)
    telemetry_est = TelemetryEstimator((proc_h, proc_w))
    corridor_engine = ApproachCorridor((proc_h, proc_w))
    hud = HUDRenderer((proc_h, proc_w), fps_target=src_fps)

    print(f"[info] Processing '{input_path}' ({src_w}x{src_h} @ {src_fps:.1f}fps, "
          f"{total_frames if total_frames > 0 else '?'} frames) "
          f"-> proc res {proc_w}x{proc_h}")
    print(f"[info] YOLO assist: {'enabled' if tracker.yolo and tracker.yolo.ok else 'disabled (fallback only)'}")

    frame_idx = 0
    t_start = time.time()
    last_report = t_start

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if scale != 1.0:
                frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

            t_sec = frame_idx / src_fps

            box, conf, state = tracker.update(frame, t_sec)
            telemetry = telemetry_est.estimate(
                tracker.last_center, tracker.last_size, tracker.confidence,
                tracker.scale_trend(), t_sec) if tracker.is_visible else None
            corridor = corridor_engine.update(tracker.last_center, tracker.last_size, t_sec)

            elapsed = time.time() - t_start
            real_fps = (frame_idx + 1) / elapsed if elapsed > 0 else src_fps
            rendered = hud.render(frame, tracker, telemetry, corridor, t_sec, real_fps=src_fps)

            writer.write(rendered)

            if telemetry_cb:
                telemetry_cb(_telemetry_record(frame_idx, t_sec, tracker, telemetry, real_fps))

            frame_idx += 1

            if progress_cb and (frame_idx % 3 == 0 or frame_idx == total_frames):
                progress_cb(frame_idx, total_frames, state, conf)

            if time.time() - last_report > 2.0:
                pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                print(f"[info] frame {frame_idx}/{total_frames if total_frames>0 else '?'} "
                      f"({pct:4.1f}%) state={state} conf={conf:.2f}")
                last_report = time.time()
    except KeyboardInterrupt:
        print("[warn] Interrupted by user, finalizing output...")
    finally:
        cap.release()
        writer.release()

    print(f"[done] Wrote {frame_idx} frames -> {out_path}")
    return out_path



def _synth_background(w, h, seed=7):

    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), dtype=np.float32)

    horizon = int(h * 0.42)

    sky_top = np.array([225, 190, 150], dtype=np.float32)
    sky_horizon = np.array([205, 195, 190], dtype=np.float32)
    for y in range(horizon):
        t = y / max(1, horizon)
        img[y, :] = sky_top * (1 - t) + sky_horizon * t

    cloud_layer = np.zeros((horizon, w), dtype=np.float32)
    for _ in range(14):
        cx = rng.uniform(0, w)
        cy = rng.uniform(0, horizon * 0.8)
        rx = rng.uniform(w * 0.08, w * 0.22)
        ry = rng.uniform(h * 0.02, h * 0.05)
        yy, xx = np.mgrid[0:horizon, 0:w]
        cloud_layer += np.exp(-(((xx - cx) ** 2) / (2 * rx ** 2) +
                                 ((yy - cy) ** 2) / (2 * ry ** 2))) * rng.uniform(15, 35)
    cloud_layer = np.clip(cloud_layer, 0, 40)
    for c in range(3):
        img[:horizon, :, c] += cloud_layer

    ground = np.zeros((h - horizon, w, 3), dtype=np.float32)
    field_h, field_w = ground.shape[0], ground.shape[1]
    base_greens = [
        np.array([70, 145, 80]), np.array([60, 130, 70]),
        np.array([85, 150, 95]), np.array([55, 120, 60]),
        np.array([95, 140, 70]),
    ]
    n_rows = 5
    row_bounds = np.linspace(0, field_h, n_rows + 1)
    for ri in range(n_rows):
        y0, y1 = int(row_bounds[ri]), int(row_bounds[ri + 1])
        depth_t = ri / max(1, n_rows - 1)
        n_cols = 3 + ri 
        col_bounds = np.linspace(0, field_w, max(2, n_cols) + 1)

        jitter = rng.uniform(-field_w * 0.02, field_w * 0.02, col_bounds.shape)
        jitter[0] = 0.0
        jitter[-1] = 0.0
        col_bounds = np.clip(col_bounds + jitter, 0, field_w)
        col_bounds.sort()
        col_bounds[0] = 0.0
        col_bounds[-1] = field_w
        for ci in range(len(col_bounds) - 1):
            x0, x1 = int(col_bounds[ci]), int(col_bounds[ci + 1])
            if x1 <= x0:
                continue
            color = base_greens[(ri + ci) % len(base_greens)].astype(np.float32)
            haze = 1.0 - depth_t * 0.35
            color = color * haze + np.array([200, 195, 185]) * (1 - haze)
            ground[y0:y1, x0:x1] = color
            n_lines = rng.integers(3, 7)
            for _ in range(n_lines):
                ly = rng.integers(y0, max(y0 + 1, y1))
                shade = rng.uniform(-14, 10)
                ground[ly:ly + 1, x0:x1] += shade

    road_top_x = w * 0.5 + rng.uniform(-w * 0.03, w * 0.03)
    road_bot_x0 = w * 0.40
    road_bot_x1 = w * 0.60
    yy, xx = np.mgrid[0:field_h, 0:field_w]
    t_row = yy / max(1, field_h - 1)
    left_edge = road_top_x + (road_bot_x0 - road_top_x) * t_row - field_w * 0.03
    right_edge = road_top_x + (road_bot_x1 - road_top_x) * t_row + field_w * 0.03
    road_mask = (xx >= left_edge) & (xx <= right_edge)
    road_color = np.array([110, 150, 165], dtype=np.float32)
    for c in range(3):
        ground[..., c] = np.where(road_mask, ground[..., c] * 0.25 + road_color[c] * 0.75,
                                    ground[..., c])

    img[horizon:, :] = ground

    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2.0, w / 2.0
    dist = np.sqrt(((xx - cx) / (w / 2)) ** 2 + ((yy - cy) / (h / 2)) ** 2)
    vignette = 1.0 - np.clip(dist - 0.55, 0, 1) * 0.35
    for c in range(3):
        img[..., c] *= vignette

    noise = rng.normal(0, 4.5, (h, w, 3))
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def run_demo(progress_cb=None, telemetry_cb=None):
    """Generate the synthetic demo perception video.

    Accepts the same optional progress_cb/telemetry_cb callbacks as
    process_video() so the dashboard API can drive and monitor demo runs
    the same way it drives real uploads. The CLI entry point (main()) does
    not pass them, so `python main.py` behaves exactly as before.
    """
    ensure_dirs()
    duration_sec = 13.0
    fps = 30
    n_frames = int(duration_sec * fps)


    target_w, target_h = 1280, 720
    bg = _synth_background(target_w, target_h)
    print("[demo] Using procedurally generated aerial scene (no reference imagery composited).")

    h, w = bg.shape[:2]
    out_path = os.path.join("output", "demo_perception.mp4")
    writer = open_writer(out_path, fps, (w, h))

  
    start = (w * 0.28, h * 0.20)
    end = (w * 0.52, h * 0.40)
    drone_start_size = (w * 0.045, w * 0.045 * 0.55)
    drone_end_size = (w * 0.085, w * 0.085 * 0.55)

    tracker = DroneTracker((h, w), use_yolo=False)
    telemetry_est = TelemetryEstimator((h, w))
    corridor_engine = ApproachCorridor((h, w))
    hud = HUDRenderer((h, w), fps_target=fps)

    rng = np.random.default_rng(42)
    wobble_x = rng.normal(0, 1, n_frames).cumsum()
    wobble_x = wobble_x / (np.max(np.abs(wobble_x)) + 1e-6)
    wobble_y = rng.normal(0, 1, n_frames).cumsum()
    wobble_y = wobble_y / (np.max(np.abs(wobble_y)) + 1e-6)

    print(f"[demo] Rendering {n_frames} frames ({duration_sec:.0f}s @ {fps}fps) -> {out_path}")

    zoom_amt = 0.06

    for i in range(n_frames):
        t = i / max(1, n_frames - 1)
        t_sec = i / fps

        search_phase = t_sec < 1.2

        te = ease(clamp((t_sec - 0.3) / (duration_sec - 0.6), 0, 1))
        cx = lerp(start[0], end[0], te) + wobble_x[i] * w * 0.015
        cy = lerp(start[1], end[1], te) + wobble_y[i] * h * 0.012
        sw = lerp(drone_start_size[0], drone_end_size[0], te)
        sh = lerp(drone_start_size[1], drone_end_size[1], te)

        zt = ease(t)
        z = 1.0 + zoom_amt * zt
        drift_x = int(w * 0.02 * math.sin(t_sec * 0.25))
        frame = _zoom_pan(bg, z, drift_x, int(h * 0.01 * math.cos(t_sec * 0.2)))

        _seed_tracker(tracker, cx, cy, sw, sh, t_sec, present=not search_phase)

        telemetry = telemetry_est.estimate(
            tracker.last_center, tracker.last_size, tracker.confidence,
            tracker.scale_trend(), t_sec) if tracker.is_visible else None
        corridor = corridor_engine.update(tracker.last_center, tracker.last_size, t_sec)

        rendered = hud.render(frame, tracker, telemetry, corridor, t_sec, real_fps=fps)
        writer.write(rendered)

        if telemetry_cb:
            telemetry_cb(_telemetry_record(i, t_sec, tracker, telemetry, fps))
        if progress_cb and (i % 2 == 0 or i == n_frames - 1):
            progress_cb(i + 1, n_frames, tracker.state, tracker.confidence)

        if i % (fps * 2) == 0:
            print(f"[demo] frame {i}/{n_frames} state={tracker.state}")

    writer.release()
    print(f"[done] Demo saved -> {out_path}")
    return out_path


def _zoom_pan(img, zoom, dx, dy):
    h, w = img.shape[:2]
    new_w, new_h = int(w * zoom), int(h * zoom)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    x0 = clamp((new_w - w) // 2 + dx, 0, max(0, new_w - w))
    y0 = clamp((new_h - h) // 2 + dy, 0, max(0, new_h - h))
    return resized[y0:y0 + h, x0:x0 + w]


def _seed_tracker(tracker, cx, cy, sw, sh, t_sec, present=True):

    if present:
        if not tracker.kf.initialized:
            tracker.kf.init(cx, cy, sw, sh)
        else:
            tracker.kf.predict()
            tracker.kf.correct(cx, cy, sw, sh)
        tracker.frames_since_hit = 0
        tracker.hits += 1
        tracker.confidence = lerp(tracker.confidence, 0.9, 0.2)
    else:
        if tracker.kf.initialized:
            tracker.kf.predict()
        tracker.frames_since_hit += 1
        tracker.confidence = lerp(tracker.confidence, 0.05, 0.15)

    if tracker.kf.initialized:
        kcx, kcy, kw, kh = (tracker.kf.kf.statePost[0], tracker.kf.kf.statePost[1],
                             tracker.kf.kf.statePost[2], tracker.kf.kf.statePost[3])
        kw = max(6.0, float(kw)); kh = max(6.0, float(kh))
        kcx, kcy = float(kcx), float(kcy)
        s_cx, s_cy = tracker.center_filter((kcx, kcy), t=t_sec)
        s_size = tracker.size_filter(math.sqrt(kw * kh), t=t_sec)
        aspect = kw / kh if kh > 0 else 1.0
        s_w = s_size * math.sqrt(aspect)
        s_h = s_size / math.sqrt(aspect)
        jump_ok = True
        if tracker.trail:
            px, py = tracker.trail[-1]
            jump_dist = math.hypot(s_cx - px, s_cy - py)
            diag = math.hypot(tracker.w, tracker.h)
            if jump_dist > diag * 0.045:
                jump_ok = False

        tracker.last_center = (s_cx, s_cy)
        tracker.last_size = (s_w, s_h)
        tracker.last_box = (s_cx - s_w / 2, s_cy - s_h / 2, s_cx + s_w / 2, s_cy + s_h / 2)
        if not jump_ok:
            tracker.trail.clear()
        min_step = math.hypot(tracker.w, tracker.h) * 0.004
        if not tracker.trail or math.hypot(
                s_cx - tracker.trail[-1][0], s_cy - tracker.trail[-1][1]) > min_step:
            tracker.trail.append(tracker.last_center)
        tracker.scale_history.append(s_w * s_h)

    tracker.age += 1
    tracker._update_state()

def main():
    ensure_dirs()
    args = sys.argv[1:]

    if len(args) >= 1:
        input_path = args[0]
        try:
            process_video(input_path)
        except Exception as e:
            print(f"[error] Processing failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        try:
            run_demo()
        except Exception as e:
            print(f"[error] Demo generation failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()