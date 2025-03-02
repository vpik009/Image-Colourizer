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
    - L2 regularization (which encourages smaller weight values by penalizing large weights)
    - The use of batches technically results in regularization due to the fact that the gradient of the loss is taken with respect to the weights of the batch rather than the entire dataset.
    - Data Augmentation 1: Rotation was included in the preprocessing of the data. The Augmented data was used as part of the training data.
    - TODO Data Augmentation 2: Multi Scale data augmentation to train the model to intake images of different sizes

## Optimizer
    - Adam optimizer is used to update the model parameters at every step while considering the exponential moving average of gradients (similar to momentum), exponential moving squared average of gradients (used to adapt the learning rate), and adaptively correcting the learing rate.

## Use of Batches
    - We use batches of size 20 to improve learning efficiency.

## Normalization
    Batch normalization is performed after every 2d convolution later.
    - Normalizes the values for each channel dimension to fit a mean of 0 and a variance of 1
    - Helps stabilize training.

## Train Data
    - TODO: switch to using ImageNet dataset for train data
    - Bing downloader was used to download the train and test data. Further details can be viewed in 'dataset.py' file
    - Downloaded images were browsed and downloaded using the following parameters:
        "animals", "people", "cars", "buildings", "trees", "mountains", "beaches", "food",
        "oceans", "rivers", "lakes", "deserts", "cities", "night", "moon", "clouds",
        "rain", "snow", "storm", "fog", "wind", "tornado", "hurricane", "earthquake", 
        "volcano", "tsunami", "fire", "ice", "water", "air", "earth", "space",
        "universe", "galaxy", "planet", "winter", "spring", "summer", "autumn", 
        "flowers", "peoples faces"
    - queries can be found in 'queries_for_download.py' file
    - Uses of data loader allows to efficiently use batches with random data samples for training.
    - Some of the data was manually filtered based on the "appropriateness" of the images for the training of the model.
    - Filtred dataset can be viewed and downloaded here: TODO

## Handling Various Image Sizes
    - Prior to the use of the downloaded images, the images were all resized to 256x256 images.
    - the transformation is performed by 'data_transformer.py' file
    - On top of resizing, the 'data_transformer.py' file performs rotational data augmentation. (View Regularization section)

## Matching the input size in the output
    Padding was used at every convolution to ensure that the activation map is the same size as the input


### Requirements
    1. torch
    2. bing-image-downloader (used to download images from BING to be used as raw training data)
    3. pathlib
    4. torchvision
    5. PIL
    6. os