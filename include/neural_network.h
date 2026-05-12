#pragma once

#include "aliases.h"
#include "dense_layer.h"
#include "matrix.h"

#include <cstdint>
#include <vector>

/// @brief Neural network for detecting handwritten digits
class NeuralNetwork {
    /// @brief layers[i] corresponds to layer $$^{(i+1)}$$
    std::vector<DenseLayer> layers;

  public:
    NeuralNetwork();

    /// @param input input datum, $$x \in [0, 1]^{28 \times 28}$$
    /// @return the last layer of activations or the prediction, $$\hat{y} \in [0, 1]^{10}$$
    Vector forward(const Vector &input);

    /// @brief Updates the weights and biases of the network
    /// @param input input datum, $$x \in [0, 1]^{28 \times 28}$$
    /// @param target true value, $$y \in \{0, 1\}^{10}$$
    /// @param learningRate gradient descent step size
    void train(const Matrix<uint8_t> &input, const Vector &target, double learningRate);
};