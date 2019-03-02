# CNN with tensorflow

Simple CNN built with tensorflow for the Kaggle competition [Plant seedlings classification](https://www.kaggle.com/c/plant-seedlings-classification).

Network architecture: <br/>

|Layers|Parameters|
|:--------:|:------------:|
|Convolution2d|filters: 32, kernel size: (5,5), activation: relu|
|MaxPooling2d|pooling size: (2,2), strides: (2,2)|
|Convolution2d|filters: 64, kernel size: (5,5), activation: relu|
|MaxPooling2d|pooling size: (2,2), strides: (2,2)|
|Convolution2d|filters: 128, kernel size: (3,3), activation: relu|
|MaxPooling2d|pooling size: (2,2), strides: (2,2)|
|FullyConnected|units: 1024, activation: relu|
|Droupout|rate: 0.35|
|FullyConnected|units: 12|

The score obtained on Kaggle (micro-averaged F1-score) is 0.73425.
This score could be increased by improving the network (adding more layers, tuning the hyperparameters).
