from imutils import paths
import argparse
import shutil
import os

ap = argparse.ArgumentParser()
#ap.add_argument("-d","--dataset", required=True,
#                 help ="path to input dataset")
#ap.add_argument("-o","--output", required=True,
#                 help ="path to output folder")
args = vars(ap.parse_args())

MAIN_PATH = "../data/chest_xray/" 
CLEAN_PATH = "../clean_data/train" 
data_folders = ["train/","test/","val/"]
labels = ["PNEUMONIA","NORMAL"]
count = 0

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
