import cv2
import time
import subprocess

# Open the camera using the native Video4Linux2 backend
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# Set the resolution explicitly (Arducam V2 works best at standard sizes)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("Starting continuous video feed...press Ctrl C to stop")

vlc_command = [
    'cvlc', '-', 
    '--sout', '#rtp{sdp=rtsp://:8554/stream}', 
    ':demux=h264'
]
vlc_process = subprocess.Popen(vlc_command, stdin=subprocess.PIPE)
print("Starting continuous video feed: To watch, open on Mac VLC with IP address")
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera")
            break
        try:
            vlc_process.stdin.write(frame.tobytes())
        except IOError:
            pass
        print("Frame captured successfully...")
        time.sleep(0.03)

except KeyboardInterrupt:
    print("Stopping video feed safely...")

finally:
    cap.release()
    if vlc_process.stdin:
        vlc_process.stdin.close()

    vlc_process.terminate()
