# neural-network.cpp

## Description

A neural network written completely from scratch. No PyTorch. No TensorFlow. Just raw C++ and the mathematics behind the magic.

The network is an implementation of the absolutely goated 3Blues1Brown series on [neural networks](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi). The network itself classifies 28 by 28 pixel handwritten single-digit numbers.

## Mathematics

### Defining the network loosely

Let $x \in [0, 1]^{28 \times 28}$ be one input datum, the input of our network
Let $\hat{y} \in [0, 1]^{10}$ be the prediction, the output of our network
Let $y \in \{0, 1\}^{10}$ be the true value

For fixed weights and biases $\theta = (W,B)$ we have
$$f_\theta: x \mapsto \hat{y}$$

It takes one input datum (e.g. one image) and outputs a prediction.

### Defining the network rigorously

Let $a_1$
