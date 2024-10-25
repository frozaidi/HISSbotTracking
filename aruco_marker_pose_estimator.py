#!/usr/bin/env python

'''
Welcome to the ArUco Marker Pose Estimator!

This program:
  - Estimates the pose of an ArUco Marker
'''

from __future__ import print_function  # Python 2/3 compatibility
import cv2  # Import the OpenCV library
import numpy as np  # Import Numpy library
from scipy.spatial.transform import Rotation as R
import math  # Math library

# Project: ArUco Marker Pose Estimator
# Date created: 12/21/2021
# Python version: 3.8

# Dictionary that was used to generate the ArUco marker
aruco_dictionary_name = "DICT_ARUCO_ORIGINAL"

# The different ArUco dictionaries built into the OpenCV library.
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
}

# Side length of the ArUco marker in meters
aruco_marker_side_length = 0.0785

# Calibration parameters yaml file
camera_calibration_parameters_filename = 'calibration_chessboard.yaml'


def euler_from_quaternion(x, y, z, w):
  t0 = +2.0 * (w * x + y * z)
  t1 = +1.0 - 2.0 * (x * x + y * y)
  roll_x = math.atan2(t0, t1)

  t2 = +2.0 * (w * y - z * x)
  t2 = +1.0 if t2 > +1.0 else t2
  t2 = -1.0 if t2 < -1.0 else t2
  pitch_y = math.asin(t2)

  t3 = +2.0 * (w * z + x * y)
  t4 = +1.0 - 2.0 * (y * y + z * z)
  yaw_z = math.atan2(t3, t4)

  return roll_x, pitch_y, yaw_z  # in radians


def main():
  if ARUCO_DICT.get(aruco_dictionary_name, None) is None:
    print("[INFO] ArUCo tag of '{}' is not supported".format(aruco_dictionary_name))
    sys.exit(0)

  cv_file = cv2.FileStorage(
      camera_calibration_parameters_filename, cv2.FILE_STORAGE_READ)
  mtx = cv_file.getNode('K').mat()
  dst = cv_file.getNode('D').mat()
  cv_file.release()

  print("[INFO] detecting '{}' markers...".format(aruco_dictionary_name))
  this_aruco_dictionary = cv2.aruco.Dictionary_get(
      ARUCO_DICT[aruco_dictionary_name])
  this_aruco_parameters = cv2.aruco.DetectorParameters_create()

  # Start the video capture from the video file
  cap = cv2.VideoCapture("aruco_videos/test.MOV")

  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      print("[INFO] End of video file reached.")
      break

    (corners, marker_ids, rejected) = cv2.aruco.detectMarkers(
        frame, this_aruco_dictionary, parameters=this_aruco_parameters,
        cameraMatrix=mtx, distCoeff=dst)

    if marker_ids is not None:
      cv2.aruco.drawDetectedMarkers(frame, corners, marker_ids)

      rvecs, tvecs, obj_points = cv2.aruco.estimatePoseSingleMarkers(
          corners,
          aruco_marker_side_length,
          mtx,
          dst)

      for i, marker_id in enumerate(marker_ids):
        transform_translation_x = tvecs[i][0][0]
        transform_translation_y = tvecs[i][0][1]
        transform_translation_z = tvecs[i][0][2]

        rotation_matrix = np.eye(4)
        rotation_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[i][0]))[0]
        r = R.from_matrix(rotation_matrix[0:3, 0:3])
        quat = r.as_quat()

        transform_rotation_x = quat[0]
        transform_rotation_y = quat[1]
        transform_rotation_z = quat[2]
        transform_rotation_w = quat[3]

        roll_x, pitch_y, yaw_z = euler_from_quaternion(transform_rotation_x,
                                                       transform_rotation_y,
                                                       transform_rotation_z,
                                                       transform_rotation_w)

        roll_x = math.degrees(roll_x)
        pitch_y = math.degrees(pitch_y)
        yaw_z = math.degrees(yaw_z)
        print("transform_translation_x: {}".format(transform_translation_x))
        print("transform_translation_y: {}".format(transform_translation_y))
        print("transform_translation_z: {}".format(transform_translation_z))
        print("roll_x: {}".format(roll_x))
        print("pitch_y: {}".format(pitch_y))
        print("yaw_z: {}".format(yaw_z))
        print()

        cv2.aruco.drawAxis(frame, mtx, dst, rvecs[i], tvecs[i], 0.05)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()


if __name__ == '__main__':
  print(__doc__)
  main()
