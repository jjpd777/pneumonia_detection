# USAGE
# python crop_accuracy.py

# import the necessary packages
from utils import config as config
from utils import SimplePreprocessor
from utils import HDF5DatasetGenerator
from utils.ranked import rank5_accuracy
from sklearn.metrics import classification_report
from keras.models import load_model
import numpy as np
import progressbar
import json
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str,
	help="path to *specific* model checkpoint to load")
args = vars(ap.parse_args())


# initialize the image preprocessors
sp = SimplePreprocessor(config.RESIZE,config.RESIZE)

test_bs = config.BATCH_SIZE//2
# load the pretrained network
print("[INFO] loading model...")
model = load_model(args["model"])

# initialize the testing dataset generator, then make predictions on
# the testing data
print("[INFO] predicting on test data (no crops)...")
testGen = HDF5DatasetGenerator(config.TEST_HDF5, config.BATCH_SIZE,
	preprocessors=[sp], classes=config.NUM_CLASSES)
predictions = model.predict_generator(testGen.generator(),
	steps=testGen.numImages //config.BATCH_SIZE , max_queue_size=10)
# compute the rank-1 and rank-5 accuracies
(rank1, _) = rank5_accuracy(predictions, testGen.db["labels"])
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
testGen.close()
