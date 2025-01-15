import cv2 as cv
import mediapipe as mp
import numpy as np



'''img_w = int(img.shape[1])
img_h = int(img.shape[0])

detector.setInputSize((img_w,img_h))

_, detections = detector.detect(img)

print(len(detections))
if detections is not None:
    for face in detections:
        x1, y1, w, h = map(int, face[:4])
        confidence = face[-1]

        cv.rectangle(img, (x1, y1), (x1+w, y1+h), (0, 255, 0), 2)

        #extract face region for mediapipe processing
        face_roi = img[y1 : y1+h , x1 : x1+w]
        rgb_face_roi = cv.cvtColor(face_roi, cv.COLOR_BGR2RGB)

        results = face_mesh.process(rgb_face_roi)     #detect landmarks

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                for landmark in landmarks.landmark:
                    x = int(landmark.x * w + x1)
                    y = int(landmark.y * h + y1)

                    cv.circle(img, (x,y), 1, (0, 0, 255), -1)'''

def DetectNLandmark(img, detector, face_mesh, profile_cascade):
    
    #use image shape to set input size for the detector
    img_w = int(img.shape[1])
    img_h = int(img.shape[0])

    detector.setInputSize((img_w,img_h))
    #YuNet detector
    _, detections = detector.detect(img)

    #Haar Cascade side profile detector
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    profile_faces = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

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

    if len(profile_faces) > 0:
        for (x, y, w, h) in profile_faces:
            
            face_data.append({"b_box": [x, y, w, h], "conf": 1.0})  # Confidence is set to 1 for Haar Cascade faces
            
            cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw bounding box for profile faces

            # Landmarking for side faces (using MediaPipe)
            face_roi = img[y:y + h, x:x + w]
            rgb_face_roi = cv.cvtColor(face_roi, cv.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_face_roi)
            if results.multi_face_landmarks:
                for landmarks in results.multi_face_landmarks:
                    landmark_coords = []
                    for landmark in landmarks.landmark:
                        x = int(landmark.x * w + x)
                        y = int(landmark.y * h + y)
                        landmark_coords.append((x, y))
                        cv.circle(img, (x, y), 1, (255, 0, 255), -1)  # Different color for profile landmarks
                    all_landmarks.append(landmark_coords)
    
    return img, face_data, all_landmarks


detector = cv.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", (320,320))

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=5, refine_landmarks=True)

profile_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_profileface.xml')

img = cv.imread('frontandside.jpg')

annotated_img, face_data, all_landmarks = DetectNLandmark(img, detector, face_mesh, profile_cascade)
print('face data:')
print(face_data)
print()
print('landmark info:')
print(all_landmarks)

#cv.imshow("detected faces", annotated_img)
#cv.waitKey(0)
#cv.destroyAllWindows()