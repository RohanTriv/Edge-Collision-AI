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

# Create a list to temporarily store your processed frames in memory
frame_buffer = []

print("Starting camera and recording frames with YOLO boxes...")
picam2.start()

# Track the exact time the loop starts
start_time = time.time()

try:
    total_frames_to_record = 1000 
    frame_count = 0
    
    print(f"Recording {total_frames_to_record} frames with YOLO integrated boxes...")
    while frame_count < total_frames_to_record:
        rgb_frame = picam2.capture_array()
        
        if rgb_frame is not None:
            bgr_canvas = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            # 1. Run YOLO tracking on the frame (Your original logic)
            results = model.track(source=bgr_canvas, conf=0.25, verbose=False)
            annotated_frame = bgr_canvas 
            
            # 2. Tell YOLO to paint the boxes, text, and labels onto the image
            if results and len(results) > 0:
                annotated_frame = results[0].plot()
            
            # 3. Store this box-filled frame into our memory buffer instead of writing it yet
            frame_buffer.append(annotated_frame)
            frame_count += 1

            if frame_count % 10 == 0:
                print(f"Captured frame {frame_count}/{total_frames_to_record}...")

    # Calculate the actual real-world processing speed
    end_time = time.time()
    total_elapsed_time = end_time - start_time
    actual_fps = frame_count / total_elapsed_time
    print(f"\nCapture finished! Total Time: {total_elapsed_time/60:.2f} minutes.")
    print(f"Your Pi processed at exactly {actual_fps:.4f} FPS.")

except KeyboardInterrupt:
    print("\nProcessing interrupted by user.")
    # Fallback calculation if you stop it early
    total_elapsed_time = time.time() - start_time
    actual_fps = len(frame_buffer) / total_elapsed_time if len(frame_buffer) > 0 else 30.0

finally:
    print("Closing camera hardware...")
    picam2.stop()
    picam2.close()
    
    # 4. Now we write the video using the CORRECT, matching real-world FPS
    if len(frame_buffer) > 0:
        print(f"Compiling video file at the real-world speed of {actual_fps:.2f} FPS...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(annotated_avi, fourcc, actual_fps, (640, 480))
        
        for frame in frame_buffer:
            video_writer.write(frame)
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
            
            if os.path.exists(annotated_avi):
                os.remove(annotated_avi)
                
            print("Done! Your video with YOLO boxes matches real-world speed perfectly.")
        except Exception as e:
            print(f"Error during MP4 conversion: {e}")
    else:
        print("No frames were captured to save.")
