import cv2
from imutils import paths
imagePaths = list(paths.list_images("./train"))

short_x = 3000
short_y = 3000
for image in imagePaths:
    a = cv2.imread(image)
    if(a is None):
        continue
    sh = a.shape
    
    if(sh[0]<short_y):
        print(sh,image)
        short_y = sh[0]
    if(sh[1]<short_x):
        print(sh,image)
        short_x = sh[1]
print(short_y,short_x)
