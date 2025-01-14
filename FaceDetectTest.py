import cv2 as cv
import mediapipe as mp
import numpy as np

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

detector = cv.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", (320,320))

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=5, refine_landmarks=True)

img = cv.imread('3faces.jpg')

img_w = int(img.shape[1])
img_h = int(img.shape[0])

detector.setInputSize((img_w,img_h))

_, detections = detector.detect(img)

print(len(detections))
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

                cv.circle(img, (x,y), 1, (0, 0, 255), -1)



cv.imshow("detected faces", img)
cv.waitKey(0)
cv.destroyAllWindows()