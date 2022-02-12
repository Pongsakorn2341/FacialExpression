from imutils import paths
import argparse
import cv2
import time
import os

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
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

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
path = 'D:\Project\\facial_cleandata\\new_image'
for imagePath in paths.list_images(args["images"]):
    image = cv2.imread(imagePath)
    dimensions = image.shape
    if(dimensions[0] != 350 or dimensions[1] != 350):
        continue
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    if fm < args["threshold"]:
        continue
    face = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(face) == 0:
        # print(imagePath)
        continue
    
    count += 1
    #print(imagePath)
    cv2.imwrite(os.path.join(path , 'grayScale'+str(count)+'.jpg'), gray)
    print(os.path.join(path , 'grayScale'+str(count)+'.jpg'))
    allImage.append(imagePath)
    # cv2.imshow('graycsale image',image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
print("Total count : ", count)
end = time.time()
print("Time range : ", end - start)
f = open("demofile2.txt", "a")
f.write("\n".join(allImage))
f.close()


# 1. filter color of image (new)
# rgb to gray scale

# 2. face detect (aom)








