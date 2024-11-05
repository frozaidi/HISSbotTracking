import cv2
import numpy as np


class ObjectTrack:
    def __init__(self):

        # Load video
        self.video = cv2.VideoCapture("aruco_videos/CCW_rev_1.mp4")


        # Define video writer to save the output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define codec
        self.output_video = cv2.VideoWriter("test_video_3.mp4", fourcc, 30.0,
                                    (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                        int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        # Define Aruco dictionary and parameters
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self.parameters = cv2.aruco.DetectorParameters()

        # Load camera calibration parameters
        camera_calibration_parameters_filename = 'calibration_chessboard.yaml'
        cv_file = cv2.FileStorage(
            camera_calibration_parameters_filename, cv2.FILE_STORAGE_READ)
        self.mtx = cv_file.getNode('K').mat()  # Camera matrix
        self.dst = cv_file.getNode('D').mat()  # Distortion coefficients
        cv_file.release()

        # Define known marker length
        self.marker_length = 0.145  # in meters

        # Target Aruco marker ID
        self.target_id = 0

    def process_video(self):
        while self.video.isOpened():
            ret, frame = self.video.read()
            if not ret:
                break
        
            self.detect_marker(frame)

        


    def detect_marker(self, frame):
        # Detect Aruco markers
        corners, ids, rejected = cv2.aruco.detectMarkers(
            frame, self.aruco_dict, parameters=self.parameters)


    def detect_color():

    def define_plane():

    def calc_point_in_world():

    def show_frame():
    
    def exit():


if __name__ == "__main__":
    tracker = ObjectTrack()