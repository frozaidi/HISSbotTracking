import cv2
import numpy as np

# Load image
image = cv2.imread("aruco_images/test3.JPG")

# Define Aruco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
parameters = cv2.aruco.DetectorParameters()

# Detect Aruco markers
corners, ids, rejected = cv2.aruco.detectMarkers(
    image, aruco_dict, parameters=parameters)

# Load camera calibration parameters
camera_calibration_parameters_filename = 'calibration_chessboard.yaml'
cv_file = cv2.FileStorage(
    camera_calibration_parameters_filename, cv2.FILE_STORAGE_READ)
mtx = cv_file.getNode('K').mat()  # Camera matrix
dst = cv_file.getNode('D').mat()  # Distortion coefficients
cv_file.release()

# Define known marker length
marker_length = 0.145  # in meters (or the actual size of your marker)

# Estimate pose of each Aruco marker
rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
    corners, marker_length, mtx, dst)

# Make sure we have at least 3 markers to define the plane
if len(tvecs) >= 3:
    # Positions of three Aruco markers in 3D space relative to the camera
    marker1_pos = tvecs[0][0]
    marker2_pos = tvecs[1][0]
    marker3_pos = tvecs[2][0]

    # Step 1: Define the plane using the three Aruco markers
    v1 = marker2_pos - marker1_pos
    v2 = marker3_pos - marker1_pos
    plane_normal = np.cross(v1, v2)
    # Normalize the normal vector
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    d = np.dot(plane_normal, marker1_pos)  # Plane equation constant

    # Find the red ball position in the image
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv_image, lower_red, upper_red)

    # Detect contours for the red ball and find its centroid
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        red_ball_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(red_ball_contour)
        red_ball_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # Step 2: Create a 3D ray from the camera to the red ball pixel
        u, v = red_ball_center
        x_normalized = (u - mtx[0, 2]) / mtx[0, 0]
        y_normalized = (v - mtx[1, 2]) / mtx[1, 1]
        ray_direction = np.array([x_normalized, y_normalized, 1.0])
        ray_direction = ray_direction / \
            np.linalg.norm(ray_direction)  # Normalize
        camera_origin = np.array([0, 0, 0])

        # Step 3: Find the intersection of the ray with the plane
        t = (d - np.dot(plane_normal, camera_origin)) / \
            np.dot(plane_normal, ray_direction)
        red_ball_3d_position = camera_origin + t * \
            ray_direction  # Intersection in 3D space
        # Check if the target marker ID 0 exists in the detected markers
        target_id = 0
        if target_id in ids:
            # Find the index of the marker with ID 0
            index = np.where(ids == target_id)[0][0]

            # Retrieve the rotation and translation vectors of the marker with ID 0
            rvec_origin = rvecs[index]
            tvec_origin = tvecs[index]

            # Step 4: Transform the red ball position to be relative to the marker with ID 0
            rotation_matrix, _ = cv2.Rodrigues(rvec_origin)
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = tvec_origin.flatten()
            inverse_transformation = np.linalg.inv(transformation_matrix)

            # Transform the red ball position
            red_ball_homogeneous = np.append(red_ball_3d_position, 1)
            red_ball_position_marker_coords = inverse_transformation @ red_ball_homogeneous
            # Extract x, y, z
            red_ball_position_marker_coords = red_ball_position_marker_coords[:3]

            print("Red ball 3D position relative to Aruco marker ID 0:",
                red_ball_position_marker_coords)
        else:
            print(f"Marker ID {target_id} not found in the image.")
else:
    print("At least three Aruco markers are needed to define the plane.")
