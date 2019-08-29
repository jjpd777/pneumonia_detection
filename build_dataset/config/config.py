# define the paths to the images directory
DATA_PATH = "../dataset"

IMAGES_PATH = DATA_PATH + "/train"

NUM_CLASSES = 2
NUM_VAL_IMAGES = 1250 * NUM_CLASSES
NUM_TEST_IMAGES = 1250 * NUM_CLASSES

TRAIN_HDF5 = "../dataset/hdf5/train.hdf5"
VAL_HDF5 = "../dataset/hdf5/val.hdf5"
TEST_HDF5 = "../dataset/hdf5/test.hdf5"

MODEL_PATH = "output/alexnet_dogs_vs_cats.model"

DATASET_MEAN = "output/dogs_vs_cats_mean.json"

OUTPUT_PATH = "output"
