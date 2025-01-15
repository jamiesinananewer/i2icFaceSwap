import cv2 as cv
import mediapipe as mp
from FaceDetectTest import DetectNLandmark

#Load YuNet face detection model
detector = cv.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", (320,320))

#load Haar cascade for side profiles
profile_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_profileface.xml')


#Load mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=20, refine_landmarks=True)

#load input video
input_video = cv.VideoCapture("snickers.mp4")


#initialise writing the output video
fourcc = cv.VideoWriter_fourcc(*'mp4v')

output_path = "snickers_side.mp4"

output_video = cv.VideoWriter(output_path, fourcc, input_video.get(cv.CAP_PROP_FPS),
                     (int(input_video.get(cv.CAP_PROP_FRAME_WIDTH)),
                      int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT))))

print(f'output video size: {(int(input_video.get(cv.CAP_PROP_FRAME_WIDTH)),
                      int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT)))}')

if not output_video.isOpened():
    print("Error: Failed to open VideoWriter.")
    exit()

framenumber = 0

#Detecting the faces and landmarks
while True:
    ret, frame = input_video.read()         #ret checks if the frame was successfully read, frame is the frame
    if not ret:
        break
    

    annotated_frame, face_data, all_landmarks = DetectNLandmark(frame, detector, face_mesh, profile_cascade)
    
    
  
    
    
    
    



    #write the frame to the output video
    if frame is not None:
        output_video.write(annotated_frame)
    else:
        print(f"Frame {framenumber} is None, skipping write.")
    framenumber += 1
    print(f"Frame {framenumber}: Processed with {len(face_data)} faces detected.")
    #print(f'frame {framenumber} processed')
    '''if framenumber == 34:
        cv.imshow('sup', frame)
        cv.waitKey(0)
        cv.destroyAllWindows()'''
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

input_video.release()
output_video.release()
