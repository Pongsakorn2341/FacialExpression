import pandas as pd
import os
import shutil
import cv2, glob, random, math, numpy as np, dlib, itertools

# Get Root Project
cwd = os.getcwd()

df = pd.read_csv(cwd + "/data/legend.csv")

df["emotion"].replace({"anger": "ANGER", "contempt": "CONTEMPT", "disgust": "DISGUST", "fear": "FEAR", \
                        "happiness": "HAPPINESS", "neutral": "NEUTRAL", "sadness": "SADNESS", "surprise": "SURPRISE"}, inplace=True)

df.drop("user_id", axis=1, inplace=True)

df_group_emotion = df.groupby("emotion").count()
print(df_group_emotion)



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(cwd + "/predictor/shape_predictor_68_face_landmarks.dat")

all_image= []
for idx, row in df.iterrows():
    
    image_path = cwd + "/images/" + row.image

    image = cv2.imread(image_path)
    height, width, channels = image.shape

    # check size image
    if(width != 350 or height != 350):
        continue

    # change color
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # check blurry
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    if fm < 5:
        continue
    
    # detect face with haarcascade
    face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_eye_tree_eyeglasses.xml')
    smile_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_smile.xml')

    face = face_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 4,
        minSize = (200, 200),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    
    for (x, y, w, h) in face:
        roi_gray = gray[y:y+h, x:x+w]

    smile = smile_cascade.detectMultiScale(
        roi_gray,
        scaleFactor = 1.16,
        minNeighbors = 35,
        minSize = (25, 25),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    eyes = eye_cascade.detectMultiScale(roi_gray)

    if len(face) != 1 or len(smile) < 1 or len(eyes) < 2:
        continue
    
    # detect face with shape predictor
    rects = detector(image, 0)

    if len(rects) == 0:
        continue

    # collect preprocessed data
    all_image.append([row.image, row.emotion])


print("Image Count : ", len(all_image))

new_df = pd.DataFrame(all_image, columns=["image", "emotion"])
new_df.to_csv(cwd + "/data_csv/preprocessing_data.csv")

