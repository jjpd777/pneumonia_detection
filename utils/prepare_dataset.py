from imutils import paths
import argparse
import shutil
import os
from pathlib import Path
import cv2
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("-m","--move_images",type=int,default=0, 
                 help ="move images to clean folder")
ap.add_argument("-c","--clean_data",type=int,default=0, 
                 help ="remove poorly formated images")
ap.add_argument("-r","--resize",type=int,default=0, 
                 help ="resize data to right proportions")
args = vars(ap.parse_args())

MAIN_PATH = "../data/chest_xray/" 
CLEAN_PATH = "../clean_data/train" 
data_folders = ["train/","test/","val/"]
labels = ["PNEUMONIA","NORMAL"]
count = 0
if args["move_images"]:
    for directory in data_folders:
        directory_in_use = MAIN_PATH + directory
        for case in labels:
            case_dir = directory_in_use + case + "/"
            print("Extracting data from ", case_dir)
            for filename in os.listdir(case_dir):
                dst = case +"-"+ str(count)+".jpeg"
                src = case_dir +filename
                dst = case_dir +dst
                os.rename(src,dst)
                count+=1
    imagePaths = list(paths.list_images(MAIN_PATH))
    print("Writing data to ", CLEAN_PATH)
    print("Total images: ",len(imagePaths))
    for image in imagePaths:
    	shutil.move(image,CLEAN_PATH)

X_MAX = 30000
Y_MAX = 30000

if args["clean_data"]:
    images = Path(CLEAN_PATH)
    images = list(images.glob("*.jpeg"))
    for image in images:
        im = cv2.imread(str(image))
        if(im is None):
            print("Faulty image",image)
            os.remove(image)
            continue
        size = im.shape
        if(size[0]<X_MAX):
            print(image,size)
            X_MAX = size[0]
            os.remove(image)
            continue
        if(size[1]<Y_MAX):
            print(image,size)
            Y_MAX = size[1]
            os.remove(image)
            continue

if args["resize"]:
    images = Path(CLEAN_PATH)
    images = list(images.glob("*.jpeg"))
    for image in tqdm(images):
        im = cv2.imread(str(image))
        im_resize = cv2.resize(im,(1200, 1500),interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(image),im_resize)




