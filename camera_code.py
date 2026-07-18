import time
import os 
import sys 
import subprocess
import cv2
import numpy as np
from picamera2 import Picamera2

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

# Configure a single YUV stream for raw processing
video_config = picam2.create_video_configuration(
    main={"format": "RGB888", "size": (640, 480)} 
)
picam2.configure(video_config)

# File paths
annotated_avi = "yolo_output.avi"
final_mp4 = "arducam_video.mp4"

# Set up the OpenCV Video Writer to record the frames WITH the bounding boxes
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(annotated_avi, fourcc, 30.0, (640, 480))

print("Starting camera and recording frames with YOLO boxes...")
picam2.start()

try:
    total_frames_to_record = 150 
    frame_count = 0
    
    print("Recording {total_frames_to_record} frames with YOLO integrated boxes...")
    while frame_count < total_frames_to_record:
        rgb_frame = picam2.capture_array()
        
        if rgb_frame is not None:
            # Extract grayscale for YOLO processing
            bgr_canvas = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            # 1. Run YOLO tracking on the frame
            results = model.track(source=bgr_canvas, conf=0.25, verbose=False)
            annotated_frame = bgr_canvas 
            # 2. Tell YOLO to paint the boxes, text, and labels onto the image
            # We convert to BGR color space here so the boxes show up in bright colors
            if results and len(results) > 0:
                annotated_frame = results[0].plot()
            
            # 3. Write this box-filled frame directly into our video file
            video_writer.write(annotated_frame)
            frame_count += 1

            if frame_count % 10 == 0:
                print(f"Captured frame {frame_count}/{total_frames_to_record}...")
            

except KeyboardInterrupt:
    print("\nProcessing interrupted by user.")

finally:
    print("Closing camera hardware and saving video stream...")
    picam2.stop()
    picam2.close()
    video_writer.release()
    
    # Convert the AVI container to a web/Mac friendly MP4 file
    print("Converting to Mac-compatible MP4 format...")
    try:
        subprocess.run([
            "ffmpeg", "-y", 
            "-i", annotated_avi, 
            "-vcodec", "libx264", 
            "-crf", "25",
            final_mp4
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Clean up temporary AVI file
        if os.path.exists(annotated_avi):
            os.remove(annotated_avi)
            
        print("Done! Your video with YOLO boxes is saved successfully as yolo_output.mp4.")
    except Exception as e:
        print(f"Error during MP4 conversion: {e}")
