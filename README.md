# Pneumonia Detection
---

This project comes from [Adrian Rosebrock](https://www.linkedin.com/in/adrian-rosebrock-59b8732a/) book _Deep Learning for Computer Vision_. For more information on the book, [take a look at this link.](https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/)

---
For this project I decided to build a deep learning solution for the [Kaggle competition of Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats). The dataset consists of a total of 25,000 images, accounting for approximately 40GB of information.

In order to speed up the training and development process, I used [this tutorial](https://www.pyimagesearch.com/2017/09/20/pre-configured-amazon-aws-deep-learning-ami-with-python/) from PyImageSearch on setting up an Amazon Machine Image (AMI) for deep learning. I used an p2.xlarge, 61 GiB Memory EC2 instance, with access to a Nvidia K80 GPU.
I downloaded the dataset directly to my AMI via the Kaggle API, and added an additional 100GB of EBS volume to work properly with the data. 


## Data Pipeline:
_Make sure you install all the requirements from the requirements.txt file_.
### 1) Extracting features using _ResNet50_:
- After downloading the dataset, I used the _ResNet50_ model from _Keras.applications_ with _ImageNet_ weights and without the fully connected layers. 
- The extracted feature vectors had shape (1x100352) and were saved in an HDF5 file. 
- To run this part of the code, execute the following on your command line:

`python extract_features.py --dataset /path/to/kaggle_dogs_vs_cats/train \
         --output /path/to/kaggle_dogs_vs_cats/hdf5/features.hdf5`
         
### 2) Training a Logistic Regression Classifier with GridSearch:
- Instead of using the fully connected layers of _ResNet50_, I trained a Logistic Regression Classifier using GridSearch to search over the **C** hyperparameter. 
- The search space for **C** was _[0.1,0.01,0.001,0.0001]_, where **0.001** yielded the best results.
- - To run this part of the code, execute the following on your command line:

`python train_model.py \
        --db ../datasets/kaggle_dogs_vs_cats/hdf5/features.hdf5 \
