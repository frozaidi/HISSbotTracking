import cv2
import numpy as np

# Load video
video = cv2.VideoCapture("aruco_videos/neon_test.MOV")

# Define Aruco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
parameters = cv2.aruco.DetectorParameters()

# Load camera calibration parameters
camera_calibration_parameters_filename = 'calibration_chessboard.yaml'
cv_file = cv2.FileStorage(
    camera_calibration_parameters_filename, cv2.FILE_STORAGE_READ)
mtx = cv_file.getNode('K').mat()  # Camera matrix
dst = cv_file.getNode('D').mat()  # Distortion coefficients
cv_file.release()

# Define known marker length
marker_length = 0.145  # in meters

# Target Aruco marker ID
target_id = 0

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # Detect Aruco markers
    corners, ids, rejected = cv2.aruco.detectMarkers(
        frame, aruco_dict, parameters=parameters)

    if ids is not None:
        # Estimate pose of each Aruco marker
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_length, mtx, dst)

        # Check for at least three markers
        if len(tvecs) >= 3:
            # Positions of three Aruco markers in 3D space relative to the camera
            marker1_pos, marker2_pos, marker3_pos = tvecs[0][0], tvecs[1][0], tvecs[2][0]

            # Define the plane using three Aruco markers
            v1 = marker2_pos - marker1_pos
            v2 = marker3_pos - marker1_pos
            plane_normal = np.cross(v1, v2)
            plane_normal = plane_normal / np.linalg.norm(plane_normal)
            d = np.dot(plane_normal, marker1_pos)

            # Detect the red ball
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_red, upper_red = np.array(
                [25, 80, 180]), np.array([35, 255, 255])
            mask = cv2.inRange(hsv_frame, lower_red, upper_red)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                red_ball_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(red_ball_contour)
                red_ball_center = (
                    int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                # 3D ray from the camera to the red ball
                u, v = red_ball_center
                x_normalized = (u - mtx[0, 2]) / mtx[0, 0]
                y_normalized = (v - mtx[1, 2]) / mtx[1, 1]
                ray_direction = np.array([x_normalized, y_normalized, 1.0])
                ray_direction /= np.linalg.norm(ray_direction)
                camera_origin = np.array([0, 0, 0])

                # Find the intersection of the ray with the plane
                t = (d - np.dot(plane_normal, camera_origin)) / \
                    np.dot(plane_normal, ray_direction)
                red_ball_3d_position = camera_origin + t * ray_direction

                # Check if the target marker ID exists
                if target_id in ids:
                    index = np.where(ids == target_id)[0][0]
                    rvec_origin, tvec_origin = rvecs[index], tvecs[index]

                    # Transform the red ball position to be relative to marker ID 0
                    rotation_matrix, _ = cv2.Rodrigues(rvec_origin)
                    transformation_matrix = np.eye(4)
                    transformation_matrix[:3, :3] = rotation_matrix
                    transformation_matrix[:3, 3] = tvec_origin.flatten()
                    inverse_transformation = np.linalg.inv(
                        transformation_matrix)

                    red_ball_homogeneous = np.append(red_ball_3d_position, 1)
                    red_ball_position_marker_coords = inverse_transformation @ red_ball_homogeneous
                    red_ball_position_marker_coords = red_ball_position_marker_coords[:3]

                    # Display the 3D coordinates
                    cv2.putText(frame, f"X: {red_ball_position_marker_coords[0]:.2f} m",
                                (red_ball_center[0] + 10,
                                 red_ball_center[1] - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame, f"Y: {red_ball_position_marker_coords[1]:.2f} m",
                                (red_ball_center[0] + 10,
                                 red_ball_center[1] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame, f"Z: {red_ball_position_marker_coords[2]:.2f} m",
                                (red_ball_center[0] + 10, red_ball_center[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # Draw a circle at the red ball's position
                    cv2.circle(frame, red_ball_center, 5, (0, 0, 255), -1)

    # Show the frame with annotations
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video and close windows
video.release()
cv2.destroyAllWindows()
