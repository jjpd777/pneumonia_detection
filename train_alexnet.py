import matplotlib
matplotlib.use("Agg")
from utils import config as config
from utils import ImageToArrayPreprocessor
from utils import SimplePreprocessor
from utils import PatchPreprocessor
from utils import MeanPreprocessor
from utils import TrainingMonitor
from utils import HDF5DatasetGenerator
from utils import AlexNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
import json
import os

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")

# initialize the image preprocessors
sp = SimplePreprocessor(config.RESIZE,config.RESIZE)

# initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, config.BATCH_SIZE, aug=aug,
	preprocessors=[sp], classes=config.NUM_CLASSES)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, config.BATCH_SIZE,
	preprocessors=[sp], classes=config.NUM_CLASSES)

# initialize the optimizer
print("[INFO] compiling model...")
opt = Adam(lr= config.DECAY, decay =config.DECAY)
model = AlexNet.build(width=config.RESIZE, height=config.RESIZE, depth=3,
	classes=config.NUM_CLASSES, reg=config.NETWORK_REG)
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])

# construct the set of callbacks
callbacks = [
	EpochCheckpoint(config.CHECKPOINTS, every=10,
		startAt=args["start_epoch"]),
	TrainingMonitor(config.MONITOR_PATH_PNG,
		jsonPath=config.MONITOR_PATH_JSON,
		startAt=args["start_epoch"]),
		LearningRateScheduler(poly_decay)]
# train the network
model.fit_generator(
	trainGen.generator(),
	steps_per_epoch=trainGen.numImages // config.BATCH_SIZE,
	validation_data=valGen.generator(),
	validation_steps=valGen.numImages // config.BATCH_SIZE,
	epochs=config.EPOCHS,
	max_queue_size=10,
	callbacks=callbacks, verbose=1)

# close the HDF5 datasets
trainGen.close()
valGen.close()
