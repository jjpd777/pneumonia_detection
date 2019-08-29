from imutils import paths
import shutil
import os

MAIN_PATH = "../datasets/chest_xray/"
CLEAN_PATH = "./clean_data/"
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
print(len(imagePaths))
im = 0 
print("Writing data to ", CLEAN_PATH)
for image in imagePaths:
	shutil.move(image,CLEAN_PATH)
	im+=1
print(im)
