import cv2
import time
import subprocess

cap = cv2.VideoCapture(0)

cap.set(cv2.CV_CAP_PROP_FRAME_WIDTH, 1280) if hasattr(cv2, 'CV_CAP_PROP_FRAME_WIDTH') else cap.set(3, 1280)
cap.set(cv2.CV_CAP_PROP_FRAME_HEIGHT, 1280) if hasattr(cv2, 'CV_CAP_PROP_FRAME_HEIGHT') else cap.set(4, 720)
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
