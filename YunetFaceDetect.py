import cv2 as cv
import mediapipe as mp

detector = cv.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", (320,320))

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=20, refine_landmarks=True)


input_video = cv.VideoCapture("TriggerGoesClick.mp4")



fourcc = cv.VideoWriter_fourcc(*'mp4v')

output_path = "jesus_detection3.mp4"

output_video = cv.VideoWriter(output_path, fourcc, input_video.get(cv.CAP_PROP_FPS),
                     (int(input_video.get(cv.CAP_PROP_FRAME_WIDTH)),
                      int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT))))


while True:
    ret, frame = input_video.read()       #ret checks if the frame was successfully read, frame is the frame
    if not ret:
        break

    frame_w = int(frame.shape[1])
    frame_h = int(frame.shape[0])

    detector.setInputSize((frame_w,frame_h))
    
    _, detections = detector.detect(frame)
    
    if detections is not None:
        for face in detections:
            x1, y1, w, h = map(int, face[:4])
            confidence = face[-1]

            cv.rectangle(frame, (x1, y1), (x1+w, y1+h), (0,255,0), 2)

            face_roi = frame[y1 : y1+h , x1 : x1+w]
            rgb_face_roi = cv.cvtColor(face_roi, cv.COLOR_BGR2RGB)

            results = face_mesh.process(rgb_face_roi)

            if results.multi_face_landmarks:
                for landmarks in results.multi_face_landmarks:
                    for landmark in landmarks.landmark:
                        x = int(landmark.x * w + x1)
                        y = int(landmark.y * h + y1)

                        cv.circle(frame, (x,y), 1, (0, 0, 255), -1)



    output_video.write(frame)
    #print('frame done')
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

input_video.release()
output_video.release()
'''
#load yunet onnx model
yunet_path = "face_detection_yunet_2023mar.onnx"

yunet= cv.dnn.readNetFromONNX(yunet_path)

#open video file
video_path = "TriggerGoesClick.mp4"

input_video = cv.VideoCapture(video_path)

#initialise OpenCV video writer
fourcc = cv.VideoWriter_fourcc(*'mp4v')

output_path = "jesus_detection.mp4"

output_video = cv.VideoWriter(output_path, fourcc, input_video.get(cv.CAP_PROP_FPS),
                     (int(input_video.get(cv.CAP_PROP_FRAME_WIDTH)),
                      int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT))))

#process each video frame
while True:
    ret, frame = input_video.read()       #ret checks if the frame was successfully read, frame is the frame
    
    if not ret:                     #if there is no frame to read, exit loop
        break
    
    #convert frame to blob for usage in yunet
    blob = cv.dnn.blobFromImage(frame, scalefactor=1.0, size=(320,320), 
                                mean=(104, 117, 123), swapRB = True, crop = False)      #converts frame from np array to 'blob'

    yunet.setInput(blob)    #set blob as input to yunet model

    #face detection
    faces = yunet.forward()

    print(f'faces = ', faces.shape)
    #loop through each detected face and draw boundary boxes
    for face in faces[0]:
        #face = face.tolist()
        #print(len(face))
        confidence = face[1]

        if confidence > 0.5:
            x1, y1, x2, y2 = (face[2:6] * [frame.shape[1], frame.shape[0], 
                                           frame.shape[1], frame.shape[0]]).astype(int)
            
            cv.rectangle(frame, (x1, y1), (x2,y2), (0, 255, 0), 2)

            label = f'Face: {confidence:.2f}'

            #cv.putText(frame, label, (x1, y1 - 10, cv.FONT_HERSHEY_SIMPLEX), 0.5, (0, 255, 0), 2)

    
    output_video.write(frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

input_video.release()
output_video.release()
            '''