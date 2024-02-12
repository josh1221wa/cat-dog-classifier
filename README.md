# Cat vs Dog Classification using CNN

## Introduction

In this notebook, we will be training an image classifier using a Convolutional Neural Network (CNN) to classify images of cats and dogs. The images used here are from various datasets in Kaggle and the model will be trained using the `tensorflow` and `keras` libraries.

This is a very simple image processing project, wherin the model just scans every image looking for patterns and isn't given any indication on where to look or any boundaries of the subject in the image

**NOTE**: This notebook, should you choose to try it for yourself, will take a pretty long time to run on a regular CPU so I would recommend using a dedicated GPU for this task.

## Datasets Used

Cats - https://www.kaggle.com/datasets/crawford/cat-dataset

Dogs - https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset

## Environment Setup

The environment details for this notebook is as follows:

- Python - 3.10.13
- CUDAtoolkit - 11.2
- cuDNN - 8.1.0

CUDA and cuDNN are required for running the program on a CUDA supported NVIDIA GPU. Check if your GPU is CUDA supported here: https://developer.nvidia.com/cuda-gpus

If you don't have a GPU, you can use the regular `tensorflow` library.

The following libraries are required to run this notebook:

- `tensorflow` - 2.10 (CUDA supported)
- `keras`
- `numpy`
- `matplotlib`
- `pandas`

## Installation

To run the code in this repository, you need to install the required dependencies. You can do this by following these steps:

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Create a conda virtual environment using the `environment.yml` file provided in the repository. This will install all the required dependencies.

```bash
conda env create -f environment.yml
```

4. Activate the environment.
5. Unzip the data files and place them in the root directory.

## Steps Followed

### 1. Image Extraction and Preprocessing

In this step, we will extract the images from the dataset and preprocess them. Any images not in the right formats will be removed. The images will be stored in batches of numpy arrays. The images will be resized to 255x255 pixels and the pixel values will be normalized to be between 0 and 1.

### 2. Model Building

In this step, we will build the CNN model using the `keras` library. The model will be trained using the training data and the validation data. The model will be compiled using the `adam` optimizer and the `binary_crossentropy` loss function.

### 3. Model Evaluation

In this step, we will test the model using the test data. We will evaluate the model using the test data and plot the accuracy and loss curves.

### 4. Model Prediction

In this step, we will use the model to make predictions on new images. We will load the model and use it to make predictions on new images.

## Conclusion

As a conclusion, I'd like to state that this is a very simple image processing project and the model is not very accurate. The model can be improved by using more advanced techniques such as transfer learning and data augmentation. However, this is a good starting point for anyone who is new to image processing and wants to learn how to build a simple image classifier using CNN.

For more details, please refer to the code comments and the generated README file. Happy coding ❤️