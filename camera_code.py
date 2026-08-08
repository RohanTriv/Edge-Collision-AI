import time
import os
import sys
import subprocess
import cv2
import numpy as np
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
# NOTE: these must match your actual wiring. Earlier in this build we wired:
#   switch = Button(5, pull_up=True)
#   led = RGBLED(red=17, green=27, blue=22)
# Update the pins below to match your physical wiring, or rewire to match these —
# pick ONE source of truth. Values below are left as originally written; change
# them to 5 / 17 / 27 / 22 (etc.) if you rewire to match the earlier test script.       
SWITCH_PIN = 5       # Input: Toggle switch pin for selecting Road Mode
GREEN_STATUS_PIN = 24 # Output: On-site verification light for TWO-WAY profile
BLUE_STATUS_PIN = 25  # Output: On-site verification light for ONE-WAY profile

if GPIO_AVAILABLE:
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LED_PIN, GPIO.OUT)
    GPIO.output(LED_PIN, GPIO.LOW)

    # Configure status confirmation pins
    GPIO.setup(GREEN_STATUS_PIN, GPIO.OUT)
    GPIO.setup(BLUE_STATUS_PIN, GPIO.OUT)

    # Configure input with internal pull-up resistor to prevent floating signals
    # Switch Open = HIGH = Two-Way Mode | Switch Closed (flipped) = LOW = One-Way Mode
    GPIO.setup(SWITCH_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

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
# Written incrementally frame-by-frame instead of buffered in a list, so memory
# use stays flat over a long-running deployment instead of growing unbounded.
# NOTE: for actual roadside deployment you'll likely want to disable continuous
# recording entirely (or add rotation/size limits) so the SD card doesn't fill up.
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
RECORDING_FPS = 15.0  # estimate; adjust to your Pi's real sustained FPS if known
video_writer = cv2.VideoWriter(annotated_avi, fourcc, RECORDING_FPS, (FRAME_W, FRAME_H))

try:
    print("Collision Radar Active. Flip the switch to confirm calibration changes.")

    # NOTE: changed from a fixed 1000-frame cap to an indefinite loop, since the
    # deployed system needs to run continuously at the roadside rather than exit
    # after ~30-60 seconds. Stop with Ctrl+C (handled below) or an external signal.
    while True:
        # 1. READ SWITCH AND TOGGLE CORRESPONDING HARDWARE STATUS INDICATORS
        if GPIO_AVAILABLE:
            if GPIO.input(SWITCH_PIN) == GPIO.LOW:
                road_mode = "one_way"
                GPIO.output(BLUE_STATUS_PIN, GPIO.HIGH)   # Turn Blue validation light ON
                GPIO.output(GREEN_STATUS_PIN, GPIO.LOW)   # Turn Green validation light OFF
            else:
                road_mode = "two_way"
                GPIO.output(GREEN_STATUS_PIN, GPIO.HIGH)  # Turn Green validation light ON
                GPIO.output(BLUE_STATUS_PIN, GPIO.LOW)    # Turn Blue validation light OFF
        else:
            road_mode = "two_way"

        rgb_frame = picam2.capture_array()
        if rgb_frame is not None:
            bgr_canvas = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            # 2. Run YOLO tracking over frame
            # FIX: model.track() returns a list of Results (one per input image),
            # even for a single frame. Index into it before using .boxes, or the
            # very first frame throws AttributeError.
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

                    # FIX: per-object wrong-way flag, so bounding box color reflects
                    # THIS object's behavior, not whichever object last tripped the
                    # frame-level flag.
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

            # Drop stale tracking history entries for objects no longer in frame,
            # so this dict doesn't grow unbounded over a long deployment.
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

            # --- HARDWARE BLINK DRIVER ---
            if collision_risk > 10:
                flash_interval = max(0.05, 0.6 - (collision_risk / 200.0))
                if time.time() - last_led_flash_time > flash_interval:
                    current_led_state = not current_led_state
                    if GPIO_AVAILABLE:
                        GPIO.output(LED_PIN, GPIO.HIGH if current_led_state else GPIO.LOW)
                    last_led_flash_time = time.time()
            else:
                if GPIO_AVAILABLE:
                    GPIO.output(LED_PIN, GPIO.LOW)

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

    # Re-encode to a more portable/compressed final mp4
    if frame_count > 0:
        subprocess.run(["ffmpeg", "-y", "-i", annotated_avi, "-vcodec", "libx264", "-crf", "25", final_mp4])