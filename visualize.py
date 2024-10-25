import cv2
import numpy as np

# Initialize Video Capture
cap = cv2.VideoCapture('IMG_9130.mov')

# Get the width, height, and frames per second (fps) of the video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
out = cv2.VideoWriter('sidewinding_cw_forward.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Define the lower and upper bounds for the yellow color in HSV
lower_yellow = np.array([28, 100, 100])
upper_yellow = np.array([40, 255, 255])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create a mask for the yellow object
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Find contours of the object
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour (assuming it's the object)
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M['m00'] > 0:
            # Calculate the centroid of the object
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            
            # Draw a marker at the centroid on the frame
            cv2.drawMarker(frame, (cX, cY), (0, 0, 255), markerType=cv2.MARKER_CROSS, 
                           markerSize=20, thickness=2, line_type=cv2.LINE_AA)
    
    # Write the frame with the overlay to the output video
    out.write(frame)
    
    # Optionally display the frame with the marker (can be removed in case of saving-only mode)
    cv2.imshow('Frame with Centroid Marker', frame)
    
    # Press 'q' to exit the loop early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video objects
cap.release()
out.release()
cv2.destroyAllWindows()
