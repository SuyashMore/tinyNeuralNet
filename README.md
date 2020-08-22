[![PR](https://camo.githubusercontent.com/f96261621753dacf526590825b84f87ccb1db0e6/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5052732d77656c636f6d652d627269676874677265656e2e7376673f7374796c653d666c6174)](https://github.com/SuyashMore/tinyNeuralNet/pulls)
[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org) 

# Tiny Neural Network

- Simple and Modular Neural Network Library built in python built using just numpy.

- Keras like Network Initialization.

- Easy to understand and build upon.

### Try the Network train and test on MNIST Dataset

**[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SuyashMore/tinyNeuralNet/blob/master/tinyNN-v2.ipynb)**

[![NN](https://cdn-images-1.medium.com/max/1000/1*Sr53EAWCc284-MDgvO3ReA.jpeg)](#) 

## Features
- Activation Functions

   - Linear

    - Sigmoid

    - Relu

    - Softmax

- Optimizer
    -  Adam

- Loss Functions
    - Cross Entropy


---

## Network Initialization 

```python
import tinyNN as tnn
nn = tnn.NeuralNetwork()
nn.addLayer(2)                       #Input Layer (2 inputs)
nn.addLayer(6,tnn.activation_sigmoid)    #Hidden Dense Layer 
nn.addLayer(6,tnn.activation_sigmoid)    #Hidden Dense Layer 
nn.addLayer(6,tnn.activation_sigmoid)    #Hidden Dense Layer 
nn.addLayer(3,tnn.activation_softmax)    #Output Layer 
nn.compile(lr=1)

# To Train 
nn.fit(Xs,Ys,epochs=5)  #Train for 5 epochs
```

---

## Installation

- Python3.6+
- numpy

### Clone

- Clone this repo to your local machine using 
```sh
git clone https://github.com/SuyashMore/tinyNeuralNet
```

---
# Implementation Details
- Weights and biases are stored as numpy Matrices


## Sample Activation Function Implementation

```python
def activation_sigmoid(X,der=False):
    if not der:
        return np.divide(1, 1 + np.exp(-X) )
    else:
        #Return Derivative of the Activation 
        return np.multiply(X,(1-X))
```

- **der** Flag Represents the derivative of the Activation Function used during BackProp

---

## Reference

- **[Andrew NG's Neural Networks and Deep Learning Course](https://www.coursera.org/learn/neural-networks-deep-learning)**

- **[3blue1brown's Neural Network Series](https://www.3blue1brown.com/neural-networks)**


## License

- **[MIT license](http://opensource.org/licenses/mit-license.php)**
- Copyright 2020 © <a href="https://github.com/SuyashMore" target="_blank">SuyashMore</a>.
