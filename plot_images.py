import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import glob

pneumonia = Path("../pneumonia-images/PNEUMONIA")
non_pneumonia = Path("../pneumonia-images/NORMAL")
pneumonia= pneumonia.glob("*.jpeg")
non_pneumonia = non_pneumonia.glob("*.jpeg")
samples = list(pneumonia) + list(non_pneumonia)


SMALL_SIZE = 12
BIG_SIZE = 30

plt.rc('axes', titlesize=BIG_SIZE)     # fontsize of the axes title
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels

f, ax = plt.subplots(2,5, figsize=(30,10))
for i in range(10):
    img = cv2.imread(str(samples[i]))
    ax[i//5, i%5].imshow(img)
    if i<5:
        ax[i//5, i%5].set_title("Pneumonia")
    else:
        ax[i//5, i%5].set_title("Normal")
    ax[i//5, i%5].axis()
    ax[i//5, i%5].set_aspect('auto')
plt.savefig("plotted_images.png")
