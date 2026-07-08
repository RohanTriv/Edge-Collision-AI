import time
import os 
import sys 
import subprocess
import cv2
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder

# Create a local tmp directory inside your project folder
local_tmp = "/home/rohan/Edge-Collision-AI/tmp"
os.makedirs(local_tmp, exist_ok=True)

# Force Python and Pip to use this new location for extractions
os.environ["TMPDIR"] = local_tmp
os.environ["PIP_TMPDIR"] = local_tmp
try:
    from ultralytics import YOLO 
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "--no-cache-dir"])
    from ultralytics import YOLO

print("Loading YOLOv8 Nano model...")
model = YOLO("yolov8n.pt") 

print("Initializing Arducam V2...")
picam2 = Picamera2()

# Configure for dual-stream: High-res video recording + Low-res array stream for YOLO
video_config = picam2.create_video_configuration(
    main={"format": "YUV420", "size": (1280, 720)}, # Saved video resolution
    lores={"format": "RGB888", "size": (640, 480)} # YOLO processing resolution (much faster!)
)
picam2.configure(video_config)

raw_filename = "arducam_video.h264"
mp4_filename = "arducam_video.mp4"

encoder = H264Encoder(bitrate=5000000) # Balanced bitrate for stability

print(f"Starting video recording... Saving to Edge-Collision-AI")
picam2.start_recording(encoder, raw_filename)

try:
    duration = 10 
    start_time = time.time()
    
    while time.time() - start_time < duration:
        # Pull the low-res background buffer frame (prevents Illegal instruction crashes)
        frame = picam2.capture_array("lores")
        
        if frame is not None:
            # BGR8888 configuration maps perfectly to OpenCV, no conversion needed!
            results = model.track(source=frame, persist=True, verbose=False)
            
            # If you want to visualize or save frames, do it here
            # annotated_frame = results[0].plot()
            
        time.sleep(0.01) # Short sleep to prevent CPU thread lock

except KeyboardInterrupt:
    print("\nProcessing interrupted by user.")

finally:
    print("Stopping recording and closing cam hardware...")
    picam2.stop_recording()
    picam2.close()
    
    print(f"Converting raw video stream to mp4...")
    try:
        subprocess.run([
            "ffmpeg", "-y", 
            "-i", raw_filename, 
            "-c:v", "copy", 
            mp4_filename
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("Done! Video with YOLO intelligence saved successfully as an MP4.")
    except Exception as e:
        print(f"Error during MP4 conversion: {e}")
