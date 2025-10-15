import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.title("👩‍🦰 Real-Time Age and Gender Detection")

# Load model files
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

# Load networks
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Lists
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

st.markdown("### Upload an image or use webcam to detect age & gender")

option = st.radio("Select input source:", ["Upload Image", "Use Webcam"])

def detect_age_gender(frame):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detections = faceNet.forward()
    h, w = frame.shape[:2]

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(face_blob)
            genderPred = genderNet.forward()
            gender = GENDER_LIST[genderPred[0].argmax()]

            ageNet.setInput(face_blob)
            agePred = ageNet.forward()
            age = AGE_LIST[agePred[0].argmax()]

            label = f"{gender}, {age}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame

if option == "Upload Image":
    img_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    if img_file is not None:
        image = Image.open(img_file)
        frame = np.array(image)
        result = detect_age_gender(frame)
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption='Result')

else:
    run = st.checkbox("Run Webcam")
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.write("Camera not found")
            break
        frame = detect_age_gender(frame)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    else:
        st.write("Stopped")
        camera.release()
