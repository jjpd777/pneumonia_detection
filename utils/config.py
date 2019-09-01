# define the paths to the images directory
IMAGES_PATH = "./clean_data/train/"

BUILD_SIZE = 700
RESIZE = 454 
NUM_CLASSES = 2
NUM_VAL_IMAGES = 300 * NUM_CLASSES
NUM_TEST_IMAGES = 300 * NUM_CLASSES

TRAIN_HDF5 = "./clean_data/hdf5/train.hdf5"
VAL_HDF5 = "./clean_data/hdf5/val.hdf5"
TEST_HDF5 = "./clean_data/hdf5/test.hdf5"

MODEL_PATH = "./output/alexnet_dogs_vs_cats.model"

DATASET_MEAN = "./output/pneumonia_mean.json"

OUTPUT_PATH = "./output"
