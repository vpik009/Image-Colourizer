# Image-Colourizer
Take a wild guess

## Model
MLP...

## Actovation Function(s)

## Loss Function
Take a wild guess

## Regularization
    1. L2 regularization (drives weights to be smaller relative to the value of the weight)
    2. Dropout

## Normalization
    Batch normalization is performed after every 2d convolution later.
    - Normalizes the values for each channel dimension to fit a mean of 0 and a variance of 1
    - Helps stabilize training, allowing higher learning rates.

## Train and Test data

## Handling Various Image Sizes ???
    1. Max pooling
    2. Global pooling to ensure the activation map before the final fully connected layer is a 1x1 image

## Matching the input size in the output
    Padding was used at every convolution to ensure that the activation map is the same size as the input


### Requirements
    1. pytorch
    2. bing-image-downloader (used for training the model)
    3. 

## Data used
    Bing downloader was used to download the train and test data. Further details can be viewed in 'dataset.py' file