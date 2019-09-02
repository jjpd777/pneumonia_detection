import cv2
from imutils import paths
import os
imagePaths = list(paths.list_images("./train"))

short_x = 30000
short_y = 30000
count = 0
for image in imagePaths:
    a = cv2.imread(image)
    if(a is None):
        os.remove(image)
        continue
    sh = a.shape
    count +=1
    if(sh[0]<short_y):
        print(sh,image)
        short_y = sh[0]
    if(sh[1]<short_x):
        print(sh,image)
        short_x = sh[1]
print(short_y,short_x)
print("Total number of good images is",count)
