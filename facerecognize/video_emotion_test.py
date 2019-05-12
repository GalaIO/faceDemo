# -*- coding: utf-8 -*-
# 
# @Author: gyj176383
# @Date: 2019/5/11
import face_recognition
import cv2
import numpy as np
import os
from keras.models import load_model

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

scan_train_path = "train_imgs"

# Create arrays of known face encodings and their names
known_face_encodings = []
known_face_names = []

if not os.path.isdir(scan_train_path):
    print('wrong scan path...')
    exit(1)

for file_name in os.listdir(scan_train_path):
    sub_path = os.path.join(scan_train_path, file_name)
    if not os.path.isdir(sub_path) and file_name.endswith('.jpg'):
        # Load a sample picture and learn how to recognize it.
        face_img = face_recognition.load_image_file(sub_path)
        face_encoding = face_recognition.face_encodings(face_img)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(file_name.split('.')[0])

# 导入性别识别 工具
gender_classifier = load_model(
    "trained_models/gender_models/simple_CNN.81-0.96.hdf5")
gender_labels = {0: 'female', 1: 'male'}

# 导入表情识别工具
emotion_classifier = load_model(
    'trained_models/emotion_models/simple_CNN.530-0.65.hdf5')
emotion_labels = {
    0: 'angry',
    1: 'disgust',
    2: 'terrified',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'calm'
}


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)

        # recognize emotion
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_face = gray[(top):(bottom), (left):(right)]
        gray_face = cv2.resize(gray_face, (48, 48))
        gray_face = gray_face / 255.0
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
        emotion = emotion_labels[emotion_label_arg]

        # recognize gender
        face = frame[(top):(bottom), (left):(right)]
        face = cv2.resize(face, (48, 48))
        face = np.expand_dims(face, 0)
        face = face / 255.0
        gender_label_arg = np.argmax(gender_classifier.predict(face))
        gender = gender_labels[gender_label_arg]

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, ' '.join((name, emotion, gender)), (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

if __name__ == '__main__':
    pass