import time
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder

print("Initializing Arducam V2...")
picam2 = Picamera2()

# Configure the camera for video recording
video_config = picam2.create_video_configuration()
picam2.configure(video_config)

# Define the output file name
encoder = H264Encoder(bitrate=10000000)  # Sets video stream quality
output_filename = "arducam_video.h264"

print(f"Starting video recording... Saving to Edge-Collision-AI")
# Start recording. The library automatically handles the encoding backend safely.
picam2.start_recording(encoder, output_filename)

try:
    # Record for 10 seconds (Change this number to record longer)
    duration = 10 
    for i in range(duration):
        print(f"Recording... {duration - i} seconds remaining.")
        time.sleep(1)

except KeyboardInterrupt:
    print("\nRecording interrupted by user.")

finally:
    # Stop recording and safely close the camera interface so it doesn't freeze
    print("Stopping recording and saving file...")
    picam2.stop_recording()
    picam2.close()
    print("Done! Video saved successfully.")
    # --- AUTOMATIC MP4 CONVERSION VIA FFMPEG ---
    print(f"Converting raw video stream to cam_mp4...")
    try:
        # Bypasses GPAC entirely and uses the native Pi video transcoder
        subprocess.run([
            "ffmpeg", "-y", 
            "-i", raw_filename, 
            "-c:v", "copy", 
            mp4_filename
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("Done! Video converted and saved successfully as an MP4.")
    except Exception as e:
        print(f"Error during MP4 conversion: {e}")
