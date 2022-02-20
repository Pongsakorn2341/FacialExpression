from cv2 import resize
from imutils import paths
import argparse
import cv2
import time
import os
import shutil

def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="./images")
ap.add_argument("-t", "--threshold", type=float, default=10.00,
	help="focus measures that fall below this value will be considered 'blurry'")
args = vars(ap.parse_args())

start = time.time()
print("Start Time : ", start)

count = 0
allImage = []

# create cleaned_images floder
path = os.getcwd()
# dir = path + '/cleaned_images'
dir = path + '/cleaned_images'
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int (w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

for imagePath in paths.list_images(args["images"]):
    image = cv2.imread(imagePath)
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
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
    smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

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

    print(len(smile))

    if len(face) == 0 or len(smile) < 1 or len(eyes) < 2:
        # print(imagePath)
        continue
    
    count += 1

    print(imagePath)
    cv2.imwrite(os.path.join(dir , (imagePath.split('\\'))[1]), gray)
    allImage.append(imagePath)
    # cv2.imshow('graycsale image',image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

print("Total count : ", count)

end = time.time()
print("Time range : ", end - start)

f = open("cleanedImage.txt", "a")
f.write("\n".join(allImage))
f.close()


# 1. filter color of image (new)
# rgb to gray scale

# 2. face detect (aom)