# define the paths to the images directory
IMAGES_PATH = "../clean_data/train/"


NUM_CLASSES = 2
NUM_VAL_IMAGES = 100 * NUM_CLASSES
NUM_TEST_IMAGES = 100 * NUM_CLASSES

TRAIN_HDF5 = "../clean_data/hdf5/train.hdf5"
VAL_HDF5 = "../clean_data/hdf5/val.hdf5"
TEST_HDF5 = "../clean_data/hdf5/test.hdf5"

MODEL_PATH = "output/alexnet_dogs_vs_cats.model"

DATASET_MEAN = "output/dogs_vs_cats_mean.json"

OUTPUT_PATH = "output"
