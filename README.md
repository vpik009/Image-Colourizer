# Image-Colourizer
    This is a project that designs and creates a small Neural Network using Convolutional Neural Networks with the purpose of learning to colourize black and white (grayscale) images.

## Model
    - Convolutional Neural Network
    - Can be viewed in 'Model_CNN.py' file

## Activation Function(s)
    - Rectified Linear Unit (ReLu)

## Loss Function
    - Mean Square Error Loss was used across batches of size 20

## Regularization
    - L2 regularization (drives weights to be smaller relative to the value of the weight)

## Optimizer
    - Adam optimizer is used to update the model parameters at every step while considering the exponential moving average of gradients (similar to momentum), exponential moving squared average of gradients (used to adapt the learning rate), and adaptively correcting the learing rate.

## Normalization?
    Batch normalization is performed after every 2d convolution later.
    - Normalizes the values for each channel dimension to fit a mean of 0 and a variance of 1
    - Helps stabilize training, allowing higher learning rates.

## Train and Test data
    - Bing downloader was used to download the train and test data. Further details can be viewed in 'dataset.py' file
    - Downloaded images were browsed and downloaded using the following parameters:
        | Query      | Limit | Adult Filter |
        |-----------|-------|--------------|
        | animals   | 100   | False        |
        | people    | 100   | False        |
        | cars      | 100   | False        |
        | buildings | 100   | False        |
        | trees     | 100   | False        |
        | mountains | 100   | False        |
        | beaches   | 100   | False        |
        | food      | 100   | False        |
        | oceans    | 100   | False        |
        | rivers    | 100   | False        |
        | lakes     | 100   | False        |
        | deserts   | 100   | False        |
        | cities    | 100   | False        |
        | night     | 100   | False        |
        | day       | 100   | False        |
        | sun       | 100   | False        |
        | moon      | 100   | False        |
        | stars     | 100   | False        |
        | clouds    | 100   | False        |
        | rain      | 100   | False        |
        | snow      | 100   | False        |
        | storm     | 100   | False        |
        | fog       | 100   | False        |
        | wind      | 100   | False        |
        | tornado   | 100   | False        |
        | hurricane | 100   | False        |
        | earthquake| 100   | False        |
        | volcano   | 100   | False        |
        | tsunami   | 100   | False        |
        | fire      | 100   | False        |
        | ice       | 100   | False        |
        | water     | 100   | False        |
        | air       | 100   | False        |
        | earth     | 100   | False        |
        | space     | 100   | False        |
        | universe  | 100   | False        |
        | galaxy    | 100   | False        |
        | planet    | 100   | False        |

    - Some of the data was manually filtered based on the "appropriateness" of the images for the training of the model.
    - Use of data loader allows to efficiently use batches with random data samples for training.

## Handling Various Image Sizes ???
    1. Max pooling
    2. Global pooling to ensure the activation map before the final fully connected layer is a 1x1 image

## Matching the input size in the output
    Padding was used at every convolution to ensure that the activation map is the same size as the input


### Requirements
    1. pytorch
    2. bing-image-downloader (used for training the model)
    3. 