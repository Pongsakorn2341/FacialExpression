from cv2 import resize
from imutils import paths
import argparse
import cv2
import time
import os
import shutil
import progressbar

def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="./utils/images")
ap.add_argument("-t", "--threshold", type=float, default=10.00,
	help="focus measures that fall below this value will be considered 'blurry'")
args = vars(ap.parse_args())

start = time.time()

count = 0
allImage = []

# create cleaned_images floder
path = os.getcwd()
# dir = path + '/cleaned_images'
dir = path + '/utils/cleaned_images'
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)

d = {'image_path': [], 'file_name': []}
path_lists = paths.list_images(args["images"])

widgets = [' [',
         progressbar.Timer(format= 'elapsed time: %(elapsed)s'),
         '] ',
           progressbar.Bar('*'),' (',
           progressbar.ETA(), ') ',
          ]
bar = progressbar.ProgressBar(max_value=13178, widgets=widgets).start()
index = 0
for image_path in path_lists:
    bar.update(index)
    index += 1
    image = cv2.imread(image_path)
    height, width, channels = image.shape

    # check size image
    if(width != 350 or height != 350):
        continue

    # change color
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    
    # check blurry
    if fm < args["threshold"]:
        continue
    
    # detect face
    face_cascade = cv2.CascadeClassifier('./utils/xml/haarcascade_frontalface_alt.xml')
    eye_cascade = cv2.CascadeClassifier('./utils/xml/haarcascade_eye_tree_eyeglasses.xml')
    smile_cascade = cv2.CascadeClassifier('./utils/xml/haarcascade_smile.xml')

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

    if len(face) == 0 or len(smile) < 1 or len(eyes) < 2:
        # print(imagePath)
        continue
    
    count += 1
    file_name = image_path.split("/")[3];
    d['image_path'].append(image_path);
    d['file_name'].append(file_name);

print("Total count : ", count)

end = time.time()
print("Time range : ", end - start)

# f = open("cleanedImage.txt", "a")
# f.write("\n".join(allImage))
# f.close()


# 1. filter color of image (new)
# rgb to gray scale

# 2. face detect (aom)
