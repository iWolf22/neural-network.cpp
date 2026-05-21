#pragma once

#include "simple/dense_layer.h"
#include "utils/aliases.h"
#include "utils/matrix.h"

#include <cstdint>
#include <initializer_list>
#include <vector>

namespace Simple {

/// @brief Neural network for detecting handwritten digits
class NeuralNetwork {
    /// @brief layers_[i] corresponds to layer $$^{(i+1)}$$
    std::vector<DenseLayer> layers_;

    void backward(const Vector &grad_lost, double learning_rate);

  public:
    /// @param layer_sizes number of neurons in each layer
    NeuralNetwork(std::initializer_list<unsigned int> layer_sizes);

    /// @param input input datum, $$x \in [0, 1]^{28 \times 28}$$
    /// @return the last layer of activations or the prediction, $$\hat{y} \in [0, 1]^{10}$$
    Vector forward(const Matrix<uint8_t> &input);

    /// @brief Updates the weights and biases of the network
    /// @param input input datum, $$x \in [0, 1]^{28 \times 28}$$
    /// @param target true value, $$y \in \{0, 1, 2, ..., 9\}$$
    /// @param learning_rate gradient descent step size
    void train(const Matrix<uint8_t> &input, uint8_t target, double learning_rate);
};

} // namespace Simple
