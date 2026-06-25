import time
import ultralytics 
from picamera2 import Picamera2
import subprocess
import cv2
from picamera2.encoders import H264Encoder
from ultralytics import YOLO

print("Loading YOLOv8 Nano model...")
# This automatically downloads the lightweight nano weights file on the first run
model = YOLO("yolov8n.pt") 

print("Initializing Arducam V2...")
picam2 = Picamera2()

# Configure the camera for video recording
video_config = picam2.create_video_configuration()
picam2.configure(video_config)

raw_filename = "arducam_video.h264"
mp4_filename = "arducam_video.mp4"

# Define the output file name
encoder = H264Encoder(bitrate=10000000)  # Sets video stream quality
output_filename = "arducam_video.h264"

print(f"Starting video recording... Saving to Edge-Collision-AI")
# Start recording. The library automatically handles the encoding backend safely.
picam2.start_recording(encoder, output_filename)

try:
    # Record for 10 seconds (Change this number to record longer)
    duration = 10 
    start_time = time.time()
    while time.time() - start_time < duration:
        frame = picam2.capture_array()
        cv2_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = model.track(source = cv2_frame, persist=True, verbose=False)
        annotated_frame = results[0].plot()
        time.sleep(0.03)

except KeyboardInterrupt:
    print("\nProcessing interrupted by user.")

finally:
    # Stop recording and safely close the camera interface so it doesn't freeze
    print("Stopping recording and closing cam hardware...")
    picam2.stop_recording()
    picam2.close()
    # --- AUTOMATIC MP4 CONVERSION VIA FFMPEG ---
    print(f"Converting raw video stream to mp4...")
    try:
        # Bypasses GPAC entirely and uses the native Pi video transcoder
        subprocess.run([
            "ffmpeg", "-y", 
            "-i", raw_filename, 
            "-c:v", "copy", 
            mp4_filename
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("Done! Video with YOLO inteligence saved successfully as an MP4.")
    except Exception as e:
        print(f"Error during MP4 conversion: {e}")
