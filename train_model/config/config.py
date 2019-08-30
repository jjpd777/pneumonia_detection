# define the paths to the images directory
IMAGES_PATH = "../clean_data/train"
IMAGE_SIZE = 400
# since we do not have validation data or access to the testing
# labels we need to take a number of images from the training
# data and use them instead
NUM_CLASSES = 2
NUM_VAL_IMAGES = 300 * NUM_CLASSES
NUM_TEST_IMAGES = 300 * NUM_CLASSES

# define the path to the output training, validation, and testing
# HDF5 files
TRAIN_HDF5 = "../clean_data/hdf5/train.hdf5"
VAL_HDF5 = "../clean_data/hdf5/val.hdf5"
TEST_HDF5 = "../clean_data/hdf5/test.hdf5"

# path to the output model file
MODEL_PATH = "output/alexnet_dogs_vs_cats.model"

# define the path to the dataset mean
DATASET_MEAN = "output/pneumonia_mean.json"

# define the path to the output directory used for storing plots,
# classification reports, etc.
OUTPUT_PATH = "output"
