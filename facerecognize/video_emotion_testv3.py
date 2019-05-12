# -*- coding: utf-8 -*-
# 
# @Author: gyj176383
# @Date: 2019/5/11
import face_recognition
import cv2
import numpy as np
import os
from keras.models import load_model

base_dir = os.path.abspath(os.path.dirname(__file__))

scan_train_path = os.path.join(base_dir, "train_imgs")

# Create arrays of known face encodings and their names
known_face_encodings = []
known_face_names = []


def load_img_from_path(dst_path):
    '''
    从目标目录加载图片
    :param dst_path:
    :return: 返回文件list  (img_name, img_path)
    '''
    if not os.path.isdir(dst_path):
        print('wrong scan path...')
        exit(1)
    files = []
    for file_name in os.listdir(dst_path):
        sub_path = os.path.join(dst_path, file_name)
        if not os.path.isdir(sub_path) and file_name.endswith('.jpg'):
            files.append((file_name, sub_path))
    return files


for img_name, img_path in load_img_from_path(scan_train_path):
    # Load a sample picture and learn how to recognize it.
    face_img = face_recognition.load_image_file(img_path)
    face_encoding = face_recognition.face_encodings(face_img)[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append(img_name.split('.')[0])

# 导入性别识别 工具
gender_classifier = load_model(
    os.path.join(base_dir, "trained_models/gender_models/simple_CNN.81-0.96.hdf5"))
gender_labels = {0: 'female', 1: 'male'}

# 导入表情识别工具
emotion_classifier = load_model(
    os.path.join(base_dir, 'trained_models/emotion_models/simple_CNN.530-0.65.hdf5'))
emotion_labels = {
    0: 'angry',
    1: 'disgust',
    2: 'terrified',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'calm'
}


def face_out(frame, factor=0.25):
    '''
    人脸识别
    :param frame:
    :return:
    '''
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=factor, fy=factor)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_persons = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
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
        mw, mh, tmp = frame.shape
        gtop = top if top - 60 < 0 else top - 60
        gbottom = bottom if bottom + 60 > mh else bottom + 60
        gleft = left if left - 30 < 0 else left - 30
        gright = right if right + 30 > mw else right + 30
        face = frame[(gtop):(gbottom), (gleft):(gright)]
        face = cv2.resize(face, (48, 48))
        face = np.expand_dims(face, 0)
        face = face / 255.0
        gender_label_arg = np.argmax(gender_classifier.predict(face))
        gender = gender_labels[gender_label_arg]

        face_persons.append((name, emotion, gender))
    convert_face_locations = []
    for (top, right, bottom, left) in face_locations:
        convert_face_locations.append((int(top / factor), int(right / factor), int(bottom / factor), int(left / factor)))
    return convert_face_locations, face_persons

live_persons = [
    {
        "name": "liying",
        "emotion": "calm",
        "gender": "male"
    },
    {
        "name": "lijiarui",
        "emotion": "calm",
        "gender": "male"
    },
    {
        "name": "guoyingjie",
        "emotion": "calm",
        "gender": "male"
    }]

def live_figout():
    '''
    人脸识别视频流实时计算
    :return:
    '''
    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)
    process_this_frame = True
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Only process every other frame of video to save time
        if process_this_frame:
            face_locations, face_persons = face_out(frame)

        process_this_frame = not process_this_frame

        # 填充全局变量
        live_persons = face_persons

        # Display the results
        for (top, right, bottom, left), person in zip(face_locations, face_persons):
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)

            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, ' '.join(person), (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


def img_figout():
    for img_name, img_path in load_img_from_path(os.path.join(base_dir, "recognize_imgs")):
        frame = cv2.imread(img_path)
        face_locations, face_persons = face_out(frame, factor=1)
        # Loop through each face found in the unknown image
        for (top, right, bottom, left), person in zip(face_locations, face_persons):
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)

            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, ' '.join(person), (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow(img_name, frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.destroyWindow(img_name)

    cv2.waitKey(0)


if __name__ == '__main__':
    live_figout()
    # img_figout()
