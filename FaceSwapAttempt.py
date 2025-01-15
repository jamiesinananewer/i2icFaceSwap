import cv2 as cv
import mediapipe as mp
import numpy as np
from FaceDetectTest import DetectNLandmark


#Load YuNet face detection model
detector = cv.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", (320,320))

#Load mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=20, refine_landmarks=True)

#load input face
user_face = cv.imread('prettyboy.jpg')

#load input video
input_video = cv.VideoCapture("snickers.mp4")


#initialise writing the output video
fourcc = cv.VideoWriter_fourcc(*'mp4v')

output_path = "snickers_new2.mp4"

output_video = cv.VideoWriter(output_path, fourcc, input_video.get(cv.CAP_PROP_FPS),
                     (int(input_video.get(cv.CAP_PROP_FRAME_WIDTH)),
                      int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT))))


#detect face and landmarks in user image
annotated_user_face, user_face_data, user_landmarks = DetectNLandmark(user_face, detector, face_mesh)

#variables for optical flow tracking
prev_gray = None
prev_landmarks = None

while True:
    ret, frame = input_video.read()         #ret checks if the frame was successfully read, frame is the frame
    if not ret:
        break
    

    #convert to grayscale for optical flow
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #use optical flow for previous landmarks
    if prev_landmarks is not None:

        prev_pts = np.array(prev_landmarks, dtype=np.float32).reshape(-1, 1, 2)

        next_pts, status, _ = cv.calcOpticalFlowPyrLK(prev_gray, gray_frame, prev_pts, None)

        #remove unsuccessfully tracked points
        tracked_landmarks = [tuple(pt.ravel()) for pt, st in zip(next_pts, status) if st == 1]





    annotated_frame, face_data, all_landmarks = DetectNLandmark(frame, detector, face_mesh)

