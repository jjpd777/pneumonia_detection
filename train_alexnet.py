import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from config import config as config
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.preprocessing import PatchPreprocessor
from pyimagesearch.preprocessing import MeanPreprocessor
from pyimagesearch.callbacks import TrainingMonitor
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.nn.conv import AlexNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
import json
import os

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")
#
# # load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())
dim = config.IMAGE_SIZE
bs = 32 
# initialize the image preprocessors
sp = SimplePreprocessor(dim,dim)
pp = PatchPreprocessor(dim,dim)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

# initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, bs, aug=aug,
	preprocessors=[pp,mp, iap], classes=2)
print("checkpoint")
valGen = HDF5DatasetGenerator(config.VAL_HDF5, bs,
	preprocessors=[sp,mp, iap], classes=2)
epochs = 50
learning_rate = 0.0017
decay = learning_rate/epochs
# initialize the optimizer
print("[INFO] compiling model...")
opt = SGD(lr=0.0018,decay=decay)
model = AlexNet.build(width=dim, height=dim, depth=3,
	classes=2, reg=0.00015)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

print("checkpoint 2 !!!!!!!!!!!!! ")
# construct the set of callbacks
path = os.path.sep.join([config.OUTPUT_PATH, "{}.png".format(
	os.getpid())])
callbacks = [TrainingMonitor(path)]

# train the network
model.fit_generator(
	trainGen.generator(),
	steps_per_epoch=trainGen.numImages // bs,
	validation_data=valGen.generator(),
	validation_steps=valGen.numImages // bs,
	epochs=epochs,
	max_queue_size=10,
	callbacks=callbacks, verbose=1)

# save the model to file
print("[INFO] serializing model...")
model.save(config.MODEL_PATH, overwrite=True)

# close the HDF5 datasets
trainGen.close()
valGen.close()
