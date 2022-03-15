from cv2 import resize
from imutils import paths
import argparse
import cv2
import time
import os
import shutil
import progressbar
from method import Method
import pandas as pd


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
bar = progressbar.ProgressBar(max_value=13719, widgets=widgets).start()
index = 0

methods = Method()
for image_path in path_lists:
    bar.update(index)
    index += 1
    image = cv2.imread(image_path)
    height, width, channels = image.shape

    if(methods.checkSizeImage(height, width) == False):
        continue;

    # change color
    gray = methods.getGrayScale(image)
    fm = methods.getVarianceLaplacian(gray)
    
    # check blurry
    if(methods.checkBlur(fm, args['threshold']) == False):
        continue

    if(methods.detectFace(gray) == False):
        continue;
    
    count += 1
    file_name = image_path.split("/")[3];
    d['image_path'].append(image_path);
    d['file_name'].append(file_name);

df = pd.DataFrame(data=d);
df.to_csv("./utils/csv/cleaned_images.csv");
print("Total count : ", count)


# f = open("cleanedImage.txt", "a")
# f.write("\n".join(allImage))
# f.close()


# 1. filter color of image (new)
# rgb to gray scale

# 2. face detect (aom)
