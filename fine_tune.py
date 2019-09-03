import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from utils import config as config
from utils import ImageToArrayPreprocessor
from utils import SimplePreprocessor
from utils import PatchPreprocessor
from utils import MeanPreprocessor
from utils import TrainingMonitor
from utils import HDF5DatasetGenerator
from utils import FCHeadNet 
from keras.applications import Xception
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input 
from keras.optimizers import Adam, SGD
from keras.models import Model
import json
import os

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")
#
# # load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())

with open('hyperparameter_tracker.txt','a') as f:
        var = str(config.BATCH_SIZE) +" | "+ str(config.LEARNING_RATE)
        f.write(var)
# initialize the image preprocessors
sp = SimplePreprocessor(config.RESIZE,config.RESIZE)
pp = PatchPreprocessor(config.RESIZE,config.RESIZE)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

# initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, config.BATCH_SIZE, aug=aug,
	preprocessors=[pp,mp, iap], classes=2)
print("checkpoint")
valGen = HDF5DatasetGenerator(config.VAL_HDF5, config.BATCH_SIZE,
	preprocessors=[sp, mp, iap], classes=2)

# initialize the optimizer
print("[INFO] compiling model...")
#opt = SGD(lr=0.0018,decay=decay)
# model = AlexNet.build(width=config.RESIZE, height=config.RESIZE, depth=3,
# 	classes=2, reg=0.00015)
baseModel = Xception(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))
headModel = FCHeadNet.build(baseModel, len(classNames), 256)
model = Model(inputs=baseModel.input, outputs=headModel)

opt = Adam(lr= config.DECAY, decay =config.DECAY)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# construct the set of callbacks
path = os.path.sep.join([config.OUTPUT_PATH, "{}.png".format(
	os.getpid())])
callbacks = [TrainingMonitor(path)]

# train the network
model.fit_generator(
	trainGen.generator(),
	steps_per_epoch=trainGen.numImages // config.BATCH_SIZE,
	validation_data=valGen.generator(),
	validation_steps=valGen.numImages // config.BATCH_SIZE,
	epochs=config.EPOCHS,
	max_queue_size=10,
	callbacks=callbacks, verbose=1)

# save the model to file
print("[INFO] serializing model...")
model.save(config.MODEL_PATH, overwrite=True)

# close the HDF5 datasets
trainGen.close()
valGen.close()
