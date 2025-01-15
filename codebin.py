

#mediapipe face detection
'''mpfd = mp.solutions.face_detection
face_detector = mpfd.FaceDetection(min_detection_confidence=0.4)

img = cv.imread("LargestSelfie.jpg")
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB) #Convert to RGB


faces = face_detector.process(img)

if faces.detections:
    for detection in faces.detections:
        bound_box = detection.location_data.relative_bounding_box

        img_h, img_w, _ = img.shape

        x1 = int(bound_box.xmin * img_w)
        y1 = int(bound_box.ymin * img_h)
        x2 = int(x1 + bound_box.width * img_w)
        y2 = int(y1 + bound_box.height * img_h)

        cv.rectangle(img, (x1,y1), (x2, y2), (255, 0, 0), 2)

cv.imshow("Mediapipe",img)
cv.waitKey(0)
cv.destroyAllWindows()'''


#detecting and landmarking video

    '''#set detector input size to be same as size as frame
    frame_w = int(frame.shape[1])           
    frame_h = int(frame.shape[0])

    detector.setInputSize((frame_w,frame_h))
    
    #detect faces in each frame using YuNet
    _, detections = detector.detect(frame)
    
    #create bounding boxes for detected faces
    if detections is not None:
        for face in detections:
            x1, y1, w, h = map(int, face[:4])
            confidence = face[-1]

            cv.rectangle(frame, (x1, y1), (x1+w, y1+h), (0,255,0), 2)       #draws the bounding box

            #Extract face region for landmark detection with mediapipe
            face_roi = frame[y1 : y1+h , x1 : x1+w]
            rgb_face_roi = cv.cvtColor(face_roi, cv.COLOR_BGR2RGB)          #converts to rgb for mediapipe

            #detect facial landmarks
            results = face_mesh.process(rgb_face_roi)

            #draw landmarks onto frame
            if results.multi_face_landmarks:
                for landmarks in results.multi_face_landmarks:
                    for landmark in landmarks.landmark:
                        x = int(landmark.x * w + x1)                        #converts landmarks into pixel coordinates
                        y = int(landmark.y * h + y1)

                        cv.circle(frame, (x,y), 1, (0, 0, 255), -1)'''


def DetectNLandmark(img, detector, face_mesh):
    
    #use image shape to set input size for the detector
    img_w = int(img.shape[1])
    img_h = int(img.shape[0])

    detector.setInputSize((img_w,img_h))

    _, detections = detector.detect(img)

    #initialise landmark and face data storage
    all_landmarks = []                              
    face_data = []                                  

    #make sure faces are detected
    if detections is not None:
        
        #iterate over each detected face
        for face in detections:
            x1, y1, w, h = map(int, face[:4])
            confidence = face[-1]

            #store bounding box and confidence level
            face_data.append({"b_box": [x1, y1, w, h] , "conf": confidence})

            #draw bounding box onto image
            cv.rectangle(img, (x1, y1), (x1+w, y1+h), (0, 255, 0), 2)

            #extract face region for mediapipe processing
            face_roi = img[y1 : y1+h , x1 : x1+w]
            rgb_face_roi = cv.cvtColor(face_roi, cv.COLOR_BGR2RGB)

            results = face_mesh.process(rgb_face_roi)     #detect landmarks

            #make sure there are landmarks detected
            if results.multi_face_landmarks:
                
                #iterate through each face's landmarks
                for landmarks in results.multi_face_landmarks:
                    landmark_coords = []
                    
                    #iterate through every individual landmark in a face
                    for landmark in landmarks.landmark:
                        
                        #map normalised landmark position to pixel co-ordinates in image
                        x = int(landmark.x * w + x1)
                        y = int(landmark.y * h + y1)

                        #store landmark co-ordinates
                        landmark_coords.append((x,y))

                        #draw landmarks onto image
                        cv.circle(img, (x,y), 1, (0, 0, 255), -1)

                    #store all landmark co-ordinates for each face
                    all_landmarks.append(landmark_coords)

    return img, face_data, all_landmarks