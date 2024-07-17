# Raga Classification using CNN

## Overview

This project aims to classify Indian music ragas using a Convolutional Neural Network (CNN). The dataset consists of audio files, each labeled with the corresponding raga. The pipeline involves preprocessing the audio data, training a CNN on Mel spectrograms of the audio files, and evaluating the model's performance.

## Dataset

The dataset consists of `.wav` audio files stored in the `/kaggle/input/indian-music-raga` directory. Each file is named with a label indicating the raga.

### Metadata Creation

A metadata CSV file is generated with the filenames and labels extracted from the audio file names. The metadata file is saved as `metadata.csv`.

## Preprocessing

### Label Encoding

The labels are encoded using `LabelEncoder` from `sklearn` and saved in an updated metadata file `updated_metadata.csv`.

### Dataset Class

A custom `Dataset` class named `Raga` is created to handle the loading and transformation of the audio files. The class includes methods for:

- Cutting or padding the audio signals to a fixed length
- Resampling the audio to a target sample rate
- Converting the audio to mono (single channel)
- Noise reduction using spectral subtraction
- Generating Mel spectrograms of the audio signals

## Model

A Convolutional Neural Network (CNN) is defined with the following architecture:

- Two convolutional layers with ReLU activation, max pooling, and dropout
- A flattening layer
- A fully connected linear layer

## Training

The training process involves:

- Initializing the dataset and data loader
- Defining the loss function and optimizer
- Training the model for a specified number of epochs

### Hyperparameters

- Batch Size: 1
- Epochs: 30
- Learning Rate: 0.001
- Sample Rate: 22050
- Number of Samples: 22050

## Evaluation

The trained model is evaluated on a test set using accuracy and classification report metrics. The predictions and expected labels are compared, and the results are summarized in a classification report.

## Results

The model achieved an accuracy of 88.89% on the test set. The classification report provides precision, recall, and f1-score for each raga class.

## Usage

To run the code, ensure that the dataset is placed in the appropriate directory and execute the script. The trained model will be saved as `raga_mel_spec.pth`. To evaluate the model, the saved state dictionary is loaded, and predictions are made on a random subset of the test set.

## Conclusion

This project demonstrates a complete pipeline for classifying Indian music ragas using deep learning. The CNN model, combined with effective preprocessing techniques, achieves high accuracy and provides a robust solution for raga classification.
