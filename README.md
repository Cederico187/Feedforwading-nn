# Feedforward neural network

## MNIST database 
The MNIST database is a well known set of 60.000 handwritten numbers as trainingsdata and another 10.000 testingdata. The numbers in the dataset look something like this: 
![alt text](images/MNIST_dataset.png)
A Feedforward neural network purpose is to learn the pattern of the given trainingsdata and trying to recognize the learned patterns when watching the testingdata.

## Architectur
My code is implememted using numpy only so i can understand the diffrent steps of maschine learning.
The neuronal network consists of one Input Layer, two Hidden Layers and an Output Layer.
The Input Layers only function is to read the dataset. Each neuron in the Input Layer represents a feature of the the input data.
Hidden Layers represent the learning function of a neuronal network and each neuron applies a weighted sum of inputs followed by a non-linear activation function. 
After the learning step is completed the now weighted neurons correspond to an final output
![alt text](images/neural_network.webp)