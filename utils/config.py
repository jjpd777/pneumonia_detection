# define the paths to the images directory
import os
import json
IMAGES_PATH = "./clean_data/train/"


BUILD_SIZE = 700
RESIZE = 300
NUM_CLASSES = 2
NUM_CHANNELS = 3
NUM_VAL_IMAGES = 300 * NUM_CLASSES
NUM_TEST_IMAGES = 300 * NUM_CLASSES

EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001
SECOND_LR =False 
MOMENTUM = False
NETWORK_REG = 0.0005
DECAY = True 
POWER = 2
#STAGES = (3,4,6)
#FILTERS = (128,64,128,256)
STAGES = False 
FILTERS =False 
FINE_TUNE_BOOL= False
FCH1 = False
FCH2 = False

TRAIN_HDF5 = "./clean_data/hdf5/train.hdf5"
VAL_HDF5 = "./clean_data/hdf5/val.hdf5"
TEST_HDF5 = "./clean_data/hdf5/test.hdf5"

EXPERIMENT_NAME = "./output/experiment-alexnet-2/"
MODEL = "AlexNet"
CHECKPOINTS = EXPERIMENT_NAME + "checkpoints/"
MONITOR_PATH_PNG = EXPERIMENT_NAME + "monitor.png"
MONITOR_PATH_JSON = EXPERIMENT_NAME + "monitor.json"
PARAMS = "parameters.txt"
PARAMS_FILE = EXPERIMENT_NAME + PARAMS
LOG_NAME = EXPERIMENT_NAME + "console.log"
MODEL_PATH = EXPERIMENT_NAME + "resnet.model"
OUTPUT_PATH = EXPERIMENT_NAME

def make_experiment():
    os.mkdir(EXPERIMENT_NAME)
    os.mkdir(CHECKPOINTS)
    os.mknod(LOG_NAME)

def store_params():
    data= {}
    data['hyperparameters'] = []
    data['hyperparameters'].append({
        'model' : MODEL,
        'image_size' : RESIZE,
        'epochs' : EPOCHS,
        'batch_size' : BATCH_SIZE,
        'learning_rate' : LEARNING_RATE,
        'second_lr': SECOND_LR,
        'stages': STAGES,
        'filters': FILTERS,
        'momentum' : MOMENTUM,
        'fine_tune' :{
            'fine_tune_used': FINE_TUNE_BOOL,
            'fc_layer_1' : FCH1,
            'fc_layer_2' : FCH2}
        })
    with open(PARAMS_FILE,'a') as write:
        json.dump(data,write)

def poly_decay(epoch):
	max_epochs = EPOCHS
	baseLR = LEARNING_RATE
	power = POWER
	alpha = baseLR * (1- (epoch / float(max_epochs)))** power
	return alpha
