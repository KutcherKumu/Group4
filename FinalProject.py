     # -*- coding: utf-8 -*-
import face_recognition
import cv2
import numpy as np
import glob
import os
import logging

IMAGES_PATH = 'images'                  #比對照片的資料夾
CAMERA_DEVICE_ID = 1                    #攝影機的編號，預設為 0，外接可以用1...之類的。
MAX_DISTANCE = 0.385                     #嚴格度，增加會降低嚴格度，減少會增加嚴格度

def get_face(image, convert_to_rgb=False): #抓到視訊裡面所有的臉和臉部特偵
    if convert_to_rgb:
        image = image[:, :, ::-1]

    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    return face_locations, face_encodings

def database():                         #載入比對照片創建臉部資訊丟到 database
    database = {}

    for filename in glob.glob(os.path.join(IMAGES_PATH, '*.jpg')):

        image_rgb = face_recognition.load_image_file(filename)

        identity = os.path.splitext(os.path.basename(filename))[0]

        locations, encodings = get_face(image_rgb)
        database[identity] = encodings[0]
        
    return database

def detected_face_on_image(frame, location, name=None):#設定畫面中邊框的文字與顏色
    top, right, bottom, left = location

    if name is None:          
        name = 'Unknown'
        color = (0, 0, 255)
    else:
        color = (0, 128, 0)
                     #加上邊框與文字
    cv2.rectangle(frame, (left, top - 35), (right, bottom), color, 2)
    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
    cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

def run(database):
    video_capture = cv2.VideoCapture(CAMERA_DEVICE_ID)      #打開攝影機

    known_face_encodings = list(database.values())
    known_face_names = list(database.keys())
    
    while video_capture.isOpened(): 
        ok, frame = video_capture.read()
        if not ok:                              #如果無法抓取就停止視訊
            logging.error("無法抓取，停止視訊")
            break

        face_locations, face_encodings = get_face(frame, convert_to_rgb=True)

        for location, face_encoding in zip(face_locations, face_encodings):

            distances = face_recognition.face_distance(known_face_encodings, face_encoding)     #比較資料庫的臉部特偵與視訊裡的臉部特徵的相似度
            if np.any(distances <= MAX_DISTANCE):               #套上與資料庫中最吻合的名子
                best_match_idx = np.argmin(distances)
                name = known_face_names[best_match_idx]
            else:                                               #沒有吻合的把name設None
                name = None
            detected_face_on_image(frame, location, name)       #如果辨識的人臉與資料中沒有吻合的 把邊框顯示的名子設成Unknown

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

database = database()
known_face_encodings = list(database.values())#抓出資料庫裡的臉部特徵資訊
known_face_names = list(database.keys())#抓出資料庫裡的臉名稱
run(database)