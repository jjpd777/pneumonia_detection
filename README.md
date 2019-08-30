# Pneumonia Detection
---

This project was built using some code from [Adrian Rosebrock](https://www.linkedin.com/in/adrian-rosebrock-59b8732a/) book _Deep Learning for Computer Vision_. For more information on the book, [take a look at this link.](https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/)

---
For this project I decided to build a deep learning solution for the [Chest X-Ray Images (Pneumonia)
](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia/activity). The dataset consists of a total of 5,860 images, accounting for approximately 10GB of information.

In order to speed up the training and development process, I used [this tutorial](https://www.pyimagesearch.com/2017/09/20/pre-configured-amazon-aws-deep-learning-ami-with-python/) from PyImageSearch on setting up an Amazon Machine Image (AMI) for deep learning. I used an p2.xlarge, 61 GiB Memory EC2 instance, with access to a Nvidia K80 GPU.
I downloaded the dataset directly to my AMI via the Kaggle API, and added an additional 100GB of EBS volume to work properly with the data. 


## Data Pipeline:
_Make sure you install all the requirements from the requirements.txt file_.
### 1) Building an _HDF5_ file to store images:
- After downloading the dataset, the first step to pre-process the data and get it in a standard format. In order to minimize the **I/O** bottleneck for _Keras_ to access the images, I first convert the images to raw binary files that are stored on the _clean_data/hdf/_ files. To build the _hdf5_ dataset, run:
`python build_hdf5.py --dataset /path/to/pneumonia/train \
         --output ../clean_data/hdf/`
### 2) Building an _HDF5_ data generator to get mini-batch "slices":
`python extract_features.py --dataset /path/to/kaggle_dogs_vs_cats/train \
         --output /path/to/kaggle_dogs_vs_cats/hdf5/features.hdf5`
         


`python train_model.py \
        --db ../datasets/kaggle_dogs_vs_cats/hdf5/features.hdf5 \
