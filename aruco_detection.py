import cv2
import numpy as np

# Set up video capture (0 for webcam or a file path for a video file)
# Replace with 0 for webcam or specify the video path
video_path = "aruco_videos/test.MOV"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise ValueError("Video not loaded. Please check the video path or URL.")

# Define the dictionary and parameters for ArUco detection
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
parameters = cv2.aruco.DetectorParameters()

# Create the ArUco detector
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Process each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the markers in the grayscale frame
    corners, ids, rejected = detector.detectMarkers(gray)
    print("Detected markers:", ids)

    # If markers are detected, draw them on the frame
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    # Display the frame with detected markers
    cv2.imshow('Detected Markers', frame)

    # Press 'q' to exit the video loop early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
