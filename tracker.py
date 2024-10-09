import cv2
import numpy as np
import matplotlib.pyplot as plt

# Initialize Video Capture
cap = cv2.VideoCapture('IMG_7936.mov')

# Get the width and height of the video frame
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the lower and upper bounds for the yellow color in HSV
lower_yellow = np.array([28, 100, 100])
upper_yellow = np.array([40, 255, 255])

# Initialize lists to store object positions over time
object_positions = []

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
            object_positions.append((cX, cY))
    
    # Optionally visualize the detection
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Convert the path to a numpy array
object_positions = np.array(object_positions)

# Set up the figure and axis for plotting
fig, ax = plt.subplots()
ax.set_xlim(0, frame_width)  # Use video frame width for x-limit
ax.set_ylim(0, frame_height)  # Use video frame height for y-limit
ax.invert_yaxis()  # Invert the y-axis
ax.set_title("Object Path with Increasing Line Width")
ax.set_xlabel("X position")
ax.set_ylabel("Y position")

# Plot the path with gradually increasing line width
num_points = len(object_positions)
for i in range(1, num_points):
    # Set a line width that gradually increases
    line_width = 1 + (i / num_points) * 2  # Start from 1 and go up to 5
    ax.plot(object_positions[i-1:i+1, 0], object_positions[i-1:i+1, 1], color='blue', linewidth=line_width)

plt.show()