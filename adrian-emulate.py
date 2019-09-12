# USAGE
# python resnet_cifar10.py --checkpoints output/checkpoints
# python resnet_cifar10.py --checkpoints output/checkpoints \
# 	--model output/checkpoints/epoch_50.hdf5 --start-epoch 50

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from utils import ImageToArrayPreprocessor
from utils import SimplePreprocessor
from utils import PatchPreprocessor
from utils import MeanPreprocessor
from utils import CropPreprocessor
from utils import HDF5DatasetGenerator
from utils import p_config as config
from utils.p_config import store_params
from utils import ResNet
from utils import EpochCheckpoint
from utils import TrainingMonitor
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.models import load_model
from keras.callbacks import LearningRateScheduler
import keras.backend as K
import numpy as np
import argparse
import sys
import json

store_params()
# set a high recursion limit so Theano doesn't complain
sys.setrecursionlimit(5000)

def poly_decay(epoch):
	max_epochs = config.EPOCHS
	baseLR = config.LEARNING_RATE
	power = config.POWER
	alpha = baseLR * (1- (epoch / float(max_epochs)))** power
	return alpha
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-c", "--checkpoints", required=True,
# 	help="path to output checkpoint directory")
ap.add_argument("-m", "--model", type=str,
	help="path to *specific* model checkpoint to load")
ap.add_argument("-s", "--start-epoch", type=int, default=0,
	help="epoch to restart training at")
args = vars(ap.parse_args())

sp = SimplePreprocessor(config.RESIZE,config.RESIZE)

aug = ImageDataGenerator(rescale= 1 / 255.0,rotation_range=20, zoom_range=0.05,
	width_shift_range=0.05, height_shift_range=0.05, shear_range=0.05,
	horizontal_flip=True, fill_mode="nearest")
valaug = ImageDataGenerator(rescale= 1 / 255.0)
# initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, config.BATCH_SIZE, aug=aug,
	preprocessors=[sp], classes=2)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, config.BATCH_SIZE, aug=valaug,
	preprocessors=[sp], classes=2)
testGen = HDF5DatasetGenerator(config.TEST_HDF5, config.BATCH_SIZE,aug=valaug,
	preprocessors=[sp], classes=config.NUM_CLASSES)

# if there is no specific model checkpoint supplied, then initialize
# the network (ResNet-56) and compile the model
if args["model"] is None:
	print("[INFO] compiling model...")
	opt = SGD(lr=config.LEARNING_RATE,decay=config.DECAY,nesterov= config.MOMENTUM)
	model = ResNet.build(config.RESIZE, config.RESIZE, 3, 2, (3,4,6),
		(64,128,256,512), reg=config.NETWORK_REG)
	model.compile(loss="binary_crossentropy", optimizer=opt,
		metrics=["accuracy"])

# otherwise, load the checkpoint from disk
else:
	print("[INFO] loading {}...".format(args["model"]))
	model = load_model(args["model"])

	# update the learning rate
	print("[INFO] old learning rate: {}".format(
		K.get_value(model.optimizer.lr)))
	K.set_value(model.optimizer.lr, 1e-5)
	print("[INFO] new learning rate: {}".format(
		K.get_value(model.optimizer.lr)))

# construct the set of callbacks
callbacks = [
	EpochCheckpoint(config.CHECKPOINTS, every=10,
		startAt=args["start_epoch"]),
	TrainingMonitor(config.EXPERIMENT_NAME+"resnet56_pneumonia.png",
		jsonPath=config.EXPERIMENT_NAME+"resnet56_pneumonia.json",
		startAt=args["start_epoch"]),
		LearningRateScheduler(poly_decay)]

# train the network
print("[INFO] training network...")
model.fit_generator(
	trainGen.generator(),
	steps_per_epoch=trainGen.numImages//config.BATCH_SIZE,
	validation_data=valGen.generator(),
	validation_steps=valGen.numImages//config.BATCH_SIZE,
	epochs=config.EPOCHS,
	max_queue_size=10,
	callbacks=callbacks, verbose=1)

trainGen.close()
valGen.close()
testGen.close()
