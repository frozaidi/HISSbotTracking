import cv2
import numpy as np
import csv
import os

path = os.getcwd()
file_list = os.listdir(path+"\\aruco_videos") 

# videofile = "CCW_rev_3.mp4"

for videofile in file_list:
    print(videofile)
    # Load video
    video = cv2.VideoCapture("aruco_videos/"+videofile)

    # Define video writer to save the output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define codec
    output_video = cv2.VideoWriter("processed_videos/processed_"+videofile, fourcc, 30.0,
                                (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                    int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))

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

    # Open CSV file for writing
    with open("processed_csv/"+videofile[:-4]+".csv", mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write header
        csv_writer.writerow(["Frame", "X", "Y", "Z"])

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            frame_number = int(video.get(cv2.CAP_PROP_POS_FRAMES))

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

                    # Define the ROI for yellow detection
                    x, y, w, h = 400, 0, 1300, 600  # Example coordinates for the region of interest
                    roi = frame[y:y+h, x:x+w]

                    # Draw a red rectangle around the ROI
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                    # Detect yellow in the ROI
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    lower_yellow, upper_yellow = np.array(
                        [20, 150, 150]), np.array([30, 255, 255])
                    yellow_mask = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)
                    contours, _ = cv2.findContours(
                        yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if contours:
                        yellow_contour = max(contours, key=cv2.contourArea)
                        M = cv2.moments(yellow_contour)

                        # Ensure M["m00"] is non-zero to avoid division by zero
                        if M["m00"] != 0:
                            yellow_center = (
                                int(M["m10"] / M["m00"]) + x, int(M["m01"] / M["m00"]) + y)

                            # Draw a circle at the detected yellow position
                            cv2.circle(frame, yellow_center, 5, (0, 255, 255), -1)

                            # Convert the 2D yellow center to a 3D point on the plane
                            u, v = yellow_center
                            x_normalized = (u - mtx[0, 2]) / mtx[0, 0]
                            y_normalized = (v - mtx[1, 2]) / mtx[1, 1]
                            ray_direction = np.array(
                                [x_normalized, y_normalized, 1.0])
                            ray_direction = ray_direction / \
                                np.linalg.norm(ray_direction)
                            camera_origin = np.array([0, 0, 0])

                            # Calculate intersection of the ray with the plane
                            t = (d - np.dot(plane_normal, camera_origin)) / \
                                np.dot(plane_normal, ray_direction)
                            yellow_ball_3d_position = camera_origin + t * ray_direction

                            # Check if marker ID 0 is detected
                            if target_id in ids:
                                index = np.where(ids == target_id)[0][0]
                                rvec_origin = rvecs[index]
                                tvec_origin = tvecs[index]

                                # Transform yellow ball's 3D position relative to marker ID 0
                                rotation_matrix, _ = cv2.Rodrigues(rvec_origin)
                                transformation_matrix = np.eye(4)
                                transformation_matrix[:3, :3] = rotation_matrix
                                transformation_matrix[:3,
                                                    3] = tvec_origin.flatten()
                                inverse_transformation = np.linalg.inv(
                                    transformation_matrix)

                                # Convert to marker 0 coordinate frame
                                yellow_ball_homogeneous = np.append(
                                    yellow_ball_3d_position, 1)
                                yellow_ball_position_marker_coords = inverse_transformation @ yellow_ball_homogeneous
                                yellow_ball_position_marker_coords = yellow_ball_position_marker_coords[
                                    :3]

                                # Write the frame number and XYZ coordinates to CSV
                                csv_writer.writerow([frame_number,
                                                    yellow_ball_position_marker_coords[0],
                                                    yellow_ball_position_marker_coords[1],
                                                    yellow_ball_position_marker_coords[2]])

                                # Display XYZ coordinates on the frame
                                label_text = f"X: {yellow_ball_position_marker_coords[0]:.2f}, " \
                                            f"Y: {yellow_ball_position_marker_coords[1]:.2f}, " \
                                            f"Z: {yellow_ball_position_marker_coords[2]:.2f}"
                                text_position = (
                                    yellow_center[0] + 10, yellow_center[1] - 10)
                                cv2.putText(frame, label_text, text_position,
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                            else:
                                print(
                                    f"Marker ID {target_id} not found in the frame.")

            # Show the frame with annotations
            # cv2.imshow("Frame", frame)
            output_video.write(frame)  # Write the frame to the output video
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the video, output video writer, and close windows
    video.release()
    output_video.release()
    cv2.destroyAllWindows()
