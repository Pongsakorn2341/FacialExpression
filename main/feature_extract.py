import pandas as pd
import numpy as np
import os
import cv2, math, dlib
from imutils import face_utils
from matplotlib import pyplot as plt

# read preprocessed data
cwd = os.getcwd()
df = pd.read_csv(cwd + "/data_csv/preprocessing_data.csv", index_col=[0])
print(df)

# ------------------------------------------------

def get_distance(fist_point, second_point):
    distance =  math.sqrt(math.pow(fist_point[0] - second_point[0], 2) + math.pow(fist_point[1] - second_point[1], 2))
    return abs(distance)

# ------------------------------------------------ 

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(cwd + "/predictor/shape_predictor_68_face_landmarks.dat")

error = []
mlist = []
distlist = []
eye_size_list = []
eye_brows_list = []

for idx, row in df.iterrows():
    image_path = cwd + "/images/" + row.image
    image = cv2.imread(image_path)

    rects = detector(image, 0)

    if len(rects) == 0:
        error.append(row.image)
    
    xlist = []
    ylist = []
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks and convert the facial landmark (x, y)
        shape = predictor(image, rect)
        shape = face_utils.shape_to_np(shape)

        # loop over coordinates, draw them on the image and store coordinates in two lists
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
            xlist.append(x)
            ylist.append(y)

    xmean = np.mean(xlist) 
    ymean = np.mean(ylist)

    # plot central face on image
    cv2.circle(image, (int(xmean), int(ymean)), 1, (0, 255, 0), -1)

    # find distance between mouth
    mavg = np.mean([ylist[61] - ylist[67], ylist[62] - ylist[66], ylist[63] - ylist[65]])
    mlist.append(mavg)

    # find distance between left eye
    left_eye_avg = np.mean([
        get_distance([xlist[37], ylist[37]], [xlist[40], ylist[40]]),
        get_distance([xlist[38], ylist[38]], [xlist[41], ylist[41]])
    ])

    # find distance between right eye
    right_eye_avg = np.mean([
        get_distance([xlist[43], ylist[43]], [xlist[46], ylist[46]]),
        get_distance([xlist[44], ylist[44]], [xlist[47], ylist[47]])
    ])
    eye_size_list.append(np.mean([left_eye_avg, right_eye_avg]))


    # find distance between eye browns
    eye_brows = np.mean([ylist[24] - ylist[26], ylist[19] - ylist[17]])
    eye_brows_list.append(eye_brows)

    # find distance between every poin to central point
    templist = []
    for i in range(17, 68):
        dist = math.sqrt(math.pow(xlist[i] - xmean, 2) + math.pow(ylist[i] - ymean, 2))
        templist.append(dist)
    distavg = np.mean(dist)
    distlist.append(distavg)
  
    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", image)# 
    k = cv2.waitKey(5) & 0xFF
    if k == 68:
        break

print("Error Counter : ", len(error))

# ------------------------------------------------

# initialize feature columns
df['mouth_distance'] = mlist
df['average_distance'] = distlist
df['eye_size'] = eye_size_list
df['eye_brows'] = eye_brows_list

df.to_csv(cwd + "/data_csv/preprocessing_data.csv", index=False)

X = df[df.columns.difference(['Unnamed: 0', 'emotion', 'image'])]
print(X.corr())


# ------------------------------------------------

# Plot PCA Graph

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
pca = PCA(n_components=4)
X_t = scaler.fit_transform(X)
pca.fit_transform(X_t)
scree_plot(X, 4, True, True, (20 , 7))
cols=['average_distance', 'eye_brows', 'eye_size', 'mouth_distance']
dpc=pd.DataFrame(pca.components_.T, 
                  index=cols, 
                  columns=X.columns)
                  
# dpc
dpc.style.applymap(lambda e: 'background-color: gray' if e > .5 else 'background-color: dark-white')

# ------------------------------------------------

import seaborn as sns
dcorr=df[cols].corr()
# dcorr

mask = np.zeros_like(dcorr)
# mask.shape
mask[np.triu_indices_from(mask)] = True

fig, ax = plt.subplots(figsize=(7,5)) 
sns.heatmap(dpc, cmap=sns.diverging_palette(10, 145, n=100), linewidths=1, 
            center=0, annot=True, vmin=-1, vmax=1)