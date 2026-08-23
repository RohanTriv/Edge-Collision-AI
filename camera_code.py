import time
import os
import sys
import subprocess
import cv2
import numpy as np
os.environ["LIBCAMERA_LOG_LEVELS"] = "*:ERROR"
from picamera2 import Picamera2

# Try importing Raspberry Pi GPIO library for physical hardware controls

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    print("[WARNING] RPi.GPIO not found. Running in simulation mode.")
    GPIO_AVAILABLE = False

# --- CONFIGURATION SETTINGS ---
FRAME_W, FRAME_H = 640, 480
VERTEX_X = 320  # Central blind curve apex point
VERTEX_Y = 240

# Designated flow direction used exclusively when switched to ONE-WAY mode
ONE_WAY_FLOW = "left_to_right"

# --- HARDWARE GPIO PIN MAP ---
# Confirmed via testing: pin 22 lights RED, pin 27 lights BLUE.
SWITCH_PIN = 5       # Input: Toggle switch pin for selecting Road Mode
BLUE_STATUS_PIN = 27 # Output: On-site verification light for TWO-WAY profile
RED_STATUS_PIN = 22  # Output: On-site verification light for ONE-WAY profile

# --- CALIBRATION CONFIRMATION SETTINGS ---
# Switch must sit still in a new position for this many seconds before the
# system actually switches over to that road mode. Prevents a brief bump or
# quick flip-through from accidentally triggering a recalibration.
CONFIRM_HOLD_SECONDS = 5.0
BLINK_HZ = 2.0  # how fast the status LED blinks while a new mode is "pending"

if GPIO_AVAILABLE:
    GPIO.setmode(GPIO.BCM)

    # Configure status confirmation pins
    GPIO.setup(RED_STATUS_PIN, GPIO.OUT)
    GPIO.setup(BLUE_STATUS_PIN, GPIO.OUT)

    # Configure input with internal pull-up resistor to prevent floating signals
    # Switch Open = HIGH = Two-Way Mode | Switch Closed (flipped) = LOW = One-Way Mode
    GPIO.setup(SWITCH_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# --- CALIBRATION STATE (persists across loop iterations) ---
def read_switch_position():
    """Returns 'one_way' or 'two_way' based on the raw switch pin, or the
    default 'two_way' if GPIO isn't available (e.g. running off-Pi)."""
    if not GPIO_AVAILABLE:
        return "two_way"
    return "one_way" if GPIO.input(SWITCH_PIN) == GPIO.LOW else "two_way"

confirmed_mode = read_switch_position()   # mode actually used by detection logic
pending_mode = confirmed_mode             # mode the switch is currently sitting on
pending_since = time.time()               # when the switch last moved to pending_mode


def set_status_led(mode, on):
    """Drives the red/blue status LEDs for the given mode. on=False turns both off."""
    if not GPIO_AVAILABLE:
        return
    if not on:
        GPIO.output(RED_STATUS_PIN, GPIO.LOW)
        GPIO.output(BLUE_STATUS_PIN, GPIO.LOW)
        return
    if mode == "one_way":
        GPIO.output(RED_STATUS_PIN, GPIO.HIGH)
        GPIO.output(BLUE_STATUS_PIN, GPIO.LOW)
    else:
        GPIO.output(BLUE_STATUS_PIN, GPIO.HIGH)
        GPIO.output(RED_STATUS_PIN, GPIO.LOW)


def update_calibration_state():
    """
    Non-blocking. Call once per main-loop iteration.
    - Reads the current switch position every call.
    - If it differs from the last confirmed mode, starts/continues a
      CONFIRM_HOLD_SECONDS countdown, blinking the corresponding status LED.
    - If the switch sits still in the new position for the full hold time,
      confirmed_mode updates and the LED goes solid.
    - If the switch moves again mid-countdown, the timer restarts against
      the newest position instead of confirming the old one.
    Returns the CURRENTLY CONFIRMED road mode (only what detection logic should use).
    """
    global confirmed_mode, pending_mode, pending_since

    current_position = read_switch_position()

    if current_position != pending_mode:
        pending_mode = current_position
        pending_since = time.time()

    elapsed = time.time() - pending_since

    if pending_mode != confirmed_mode:
        # Counting down toward a new mode -> blink to show "confirming..."
        blink_on = int(elapsed * BLINK_HZ * 2) % 2 == 0
        set_status_led(pending_mode, blink_on)

        if elapsed >= CONFIRM_HOLD_SECONDS:
            confirmed_mode = pending_mode
            set_status_led(confirmed_mode, True)
            print(f"Road mode CONFIRMED: {confirmed_mode.upper()}")
    else:
        # Stable, already confirmed -> solid LED, no blinking
        set_status_led(confirmed_mode, True)

    return confirmed_mode


# System tracking cache
tracking_history = {}
last_led_flash_time = time.time()
current_led_state = False

# How many frames a tracked object can go unseen before we drop it from history.
# Prevents tracking_history from growing without bound during long deployments.
TRACK_STALE_FRAMES = 90  # ~3 seconds at 30fps

local_tmp = "/home/rohan/Edge-Collision-AI/tmp"
os.makedirs(local_tmp, exist_ok=True)
os.environ["TMPDIR"] = local_tmp
os.environ["PIP_TMPDIR"] = local_tmp

try:
    import logging
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    from ultralytics import YOLO
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "--no-cache-dir"])
    from ultralytics import YOLO

print("Loading YOLOv8 Nano model...")
model = YOLO("yolov8n.pt")

print("Initializing Arducam V2 Radar Array...")
picam2 = Picamera2()
video_config = picam2.create_video_configuration(main={"format": "RGB888", "size": (FRAME_W, FRAME_H)})
picam2.configure(video_config)

annotated_avi = "yolo_output.avi"
final_mp4 = "arducam_video.mp4"

# Filtered COCO IDs: 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
road_safety_classes = [0, 1, 2, 3, 5, 7]

picam2.start()
start_time = time.time()
frame_count = 0

# --- VIDEO WRITER SETUP ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
RECORDING_FPS = 15.0  # estimate; adjust to your Pi's real sustained FPS if known
video_writer = cv2.VideoWriter(annotated_avi, fourcc, RECORDING_FPS, (FRAME_W, FRAME_H))

try:
    print(f"Collision Radar Active. Hold the switch in a position for {CONFIRM_HOLD_SECONDS:.0f}s to change calibration.")

    while True:
        # 1. UPDATE CALIBRATION STATE (non-blocking hold-to-confirm logic)
        road_mode = update_calibration_state()

        rgb_frame = picam2.capture_array()
        if rgb_frame is not None:
            bgr_canvas = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            # 2. Run YOLO tracking over frame
            track_results = model.track(source=bgr_canvas, conf=0.25, persist=True, verbose=False, classes=road_safety_classes)
            results = track_results[0]
            annotated_frame = bgr_canvas.copy()

            # Track evaluation registers
            left_oncoming = []
            right_oncoming = []
            wrong_way_detected = False   # frame-level flag, used for the risk score
            pedestrian_hazard = False

            seen_ids_this_frame = set()

            if results.boxes is not None and results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                track_ids = results.boxes.id.cpu().numpy().astype(int)
                class_ids = results.boxes.cls.cpu().numpy().astype(int)

                # Visual Anchor Points
                cv2.circle(annotated_frame, (VERTEX_X, VERTEX_Y), 15, (0, 0, 255), -1)

                for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
                    seen_ids_this_frame.add(track_id)
                    x1, y1, x2, y2 = box
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    is_pedestrian = (cls_id == 0)

                    distance_to_vertex = np.sqrt((cx - VERTEX_X) ** 2 + (cy - VERTEX_Y) ** 2)
                    speed_px = 0.0
                    prev_x = cx

                    this_object_wrong_way = False

                    if track_id in tracking_history:
                        prev_x, prev_y = tracking_history[track_id]["last_pos"]
                        speed_px = np.sqrt((cx - prev_x) ** 2 + (cy - prev_y) ** 2)
                        speed_px = (0.7 * speed_px) + (0.3 * tracking_history[track_id]["speed_px_per_frame"])

                    tracking_history[track_id] = {
                        "last_pos": (cx, cy),
                        "speed_px_per_frame": speed_px,
                        "last_seen_frame": frame_count,
                    }

                    if speed_px > 0.5:
                        tta_frames = distance_to_vertex / speed_px

                        if road_mode == "two_way":
                            if cx < VERTEX_X and cx > prev_x:
                                left_oncoming.append({"id": track_id, "tta": tta_frames})
                            elif cx > VERTEX_X and cx < prev_x:
                                right_oncoming.append({"id": track_id, "tta": tta_frames})

                        elif road_mode == "one_way":
                            if ONE_WAY_FLOW == "left_to_right":
                                if cx < prev_x and not is_pedestrian:
                                    this_object_wrong_way = True
                                    wrong_way_detected = True
                            elif ONE_WAY_FLOW == "right_to_left":
                                if cx > prev_x and not is_pedestrian:
                                    this_object_wrong_way = True
                                    wrong_way_detected = True

                    if is_pedestrian and distance_to_vertex < 150:
                        pedestrian_hazard = True

                    box_color = (0, 0, 255) if this_object_wrong_way else (0, 255, 0)
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)

            # Drop stale tracking history entries for objects no longer in frame
            stale_ids = [
                tid for tid, data in tracking_history.items()
                if frame_count - data.get("last_seen_frame", frame_count) > TRACK_STALE_FRAMES
            ]
            for tid in stale_ids:
                del tracking_history[tid]

            # --- PROCESS RISK ALGORITHMS ---
            collision_risk = 0.0
            status_text = "ROAD CLEAR"

            if road_mode == "two_way":
                if left_oncoming and right_oncoming:
                    closest_left = min(left_oncoming, key=lambda x: x["tta"])
                    closest_right = min(right_oncoming, key=lambda x: x["tta"])
                    tta_delta = abs(closest_left["tta"] - closest_right["tta"])
                    if tta_delta < 45:
                        collision_risk = max(0.0, min(100.0, (1.0 - (tta_delta / 45.0)) * 100.0))
                        status_text = "CROSS TRAFFIC THREAT"

            elif road_mode == "one_way":
                if wrong_way_detected:
                    collision_risk = 100.0
                    status_text = "WRONG WAY VEHICLE DETECTED"
                elif pedestrian_hazard:
                    collision_risk = 75.0
                    status_text = "PEDESTRIAN IN BLIND CURVE"

            # --- ON-SCREEN VIDEO BANNER CONFIRMATION OVERLAY ---
            banner_color = (0, 128, 0) if road_mode == "two_way" else (139, 0, 0)  # Green bar vs Blue/Navy bar
            cv2.rectangle(annotated_frame, (0, 0), (FRAME_W, 35), banner_color, -1)

            banner_text = f"CALIBRATION PROFILE: {road_mode.upper()} MODE ACTIVE"
            cv2.putText(annotated_frame, banner_text, (15, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

            risk_color = (0, 255, 0) if collision_risk < 30 else ((0, 165, 255) if collision_risk < 70 else (0, 0, 255))
            cv2.putText(annotated_frame, f"RISK ASSESSMENT: {collision_risk:.1f}%", (20, 65),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, risk_color, 2)
            cv2.putText(annotated_frame, f"STATUS: {status_text}", (20, 95),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

            # --- HARDWARE BLINK DRIVER (warning strip - not yet wired) ---
            #  if collision_risk > 10:
            #     flash_interval = max(0.05, 0.6 - (collision_risk / 200.0))
            #     if time.time() - last_led_flash_time > flash_interval:
            #        current_led_state = not current_led_state
            #        if GPIO_AVAILABLE:
            #           GPIO.output(LED_PIN, GPIO.HIGH if current_led_state else GPIO.LOW)
            #       last_led_flash_time = time.time()
            # else:
            #   if GPIO_AVAILABLE:
            #    GPIO.output(LED_PIN, GPIO.LOW)

            # Write this frame directly instead of buffering in a list
            video_writer.write(annotated_frame)
            frame_count += 1

except KeyboardInterrupt:
    print("\nSystem paused.")
finally:
    if GPIO_AVAILABLE:
        GPIO.cleanup()
    picam2.stop()
    picam2.close()
    video_writer.release()
    elapsed_real_time = time.time() - start_time
    actual_fps = frame_count / elapsed_real_time if elapsed_real_time > 0 else 15.0
    print(f"[INFO] Captured {frame_count} frames in {elapsed_real_time:.1f}s real time -> actual FPS: {actual_fps:.2f}")
    # Re-encode to a more portable/compressed final mp4
    if frame_count > 0:
        subprocess.run(["ffmpeg", "-y", "-i", annotated_avi, "-vcodec", "libx264", "-crf", "25", final_mp4])