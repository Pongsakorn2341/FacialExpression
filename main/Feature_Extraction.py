import pandas as pd, numpy as np, os, cv2, math, dlib
import seaborn as sns
from imutils import face_utils


cwd = os.getcwd()
pre_path = "/data_csv/preprocessing_data.csv"
df = pd.read_csv(cwd + pre_path)

def console_main(name = ""):
    print('''
        * ----------''', name,''' -------- *
    ''')
def get_distance(fist_point, second_point):
    distance =  math.sqrt(math.pow(fist_point[0] - second_point[0], 2) + math.pow(fist_point[1] - second_point[1], 2))
    return abs(distance)

console_main("Emotion List")
print(df.groupby('emotion').count())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(cwd + "/predictor/shape_predictor_68_face_landmarks.dat")

error = []
mlist = []
distlist = []
eye_size_list = []
eye_brows_list = []


for idx, row  in df.iterrows():
    image_path = cwd + "/images/" + row.image
    image = cv2.imread(image_path)

    if(len(rects) == 0):
        continue

    xlist, ylist = [], []
    for (i, rect) in enumerate(rects):
        shape = face_utils.shape_to_np(predictor(image, rect))

        for (x, y) in shape:
            xlist.append(x)
            ylist.append(y)

    xmean, ymean = np.mean(xlist), np.mean(ylist)
    
    if(not xmean or not ymean):
        continue

    # 1. find distance between mouth
    mavg = np.mean([ylist[61] - ylist[67], ylist[62] - ylist[66], ylist[63] - ylist[65]])
    mlist.append(mavg)

    # 2. find mean of left_eye & right_eye 
    left_eye_avg = np.mean([
        get_distance([xlist[37], ylist[37]], [xlist[40], ylist[40]]),
        get_distance([xlist[38], ylist[38]], [xlist[41], ylist[41]])
    ])
    right_eye_avg = np.mean([
        get_distance([xlist[43], ylist[43]], [xlist[46], ylist[46]]),
        get_distance([xlist[44], ylist[44]], [xlist[47], ylist[47]])
    ])
    eye_size_list.append(np.mean([left_eye_avg, right_eye_avg]))

    # 3. find distance between eye browns
    eye_brows = np.mean([ylist[24] - ylist[26], ylist[19] - ylist[17]])
    eye_brows_list.append(eye_brows)

    # 4. find distance between every poin to central point
    templist = []
    for i in range(17, 68):
        dist = math.sqrt(math.pow(xlist[i] - xmean, 2) + math.pow(ylist[i] - ymean, 2))
        templist.append(dist)
    distavg = np.mean(dist)
    distlist.append(distavg)


console_main("Error Counter")
print("Counter : ", len(error))

console_main("Initialize Feature")
df['mouth_distance'] = mlist
df['average_distance'] = distlist
df['eye_size'] = eye_size_list
df['eye_brows'] = eye_brows_list


X, y = df[df.columns.difference(['Unnamed: 0', 'emotion', 'image'])], df[['emotion']]




