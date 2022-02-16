from PIL import Image
import os
import PIL
import glob
import time
import cv2
from imutils import paths
import argparse
import shutil
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="./images")
args = vars(ap.parse_args())
count = 0
allImage = []
path = os.getcwd()
dir = path + '/cleaned_images'
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)
fixed_height = 350
for imagePath in paths.list_images(args["images"]):
    image = Image.open(imagePath)
    height_percent = (fixed_height / float(image.size[1]))
    width_size = int((float(image.size[0]) * float(height_percent)))
    image = image.resize((width_size, fixed_height), PIL.Image.NEAREST)
    imageName = (imagePath.split("/"))[1]
    image.save(imageName)