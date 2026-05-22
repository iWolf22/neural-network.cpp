#pragma once

#include "batched/dense_layer.h"
#include "utils/aliases.h"
#include "utils/matrix.h"

#include <cstdint>
#include <initializer_list>
#include <vector>

namespace Batched {

/// @brief Neural network for detecting handwritten digits
class NeuralNetwork {
    /// @brief layers_[i] corresponds to layer $$^{(i+1)}$$
    std::vector<DenseLayer> layers_;

    void backward(const Matrix<double> &grad_lost);

  public:
    const double learning_rate_;
    const size_t batch_size_;

    /// @param learning_rate gradient descent step size
    /// @param layer_sizes number of neurons in each layer
    NeuralNetwork(double learning_rate, size_t batch_size, std::initializer_list<unsigned int> layer_sizes);

    /// @param input several input datum in a single batch, $$x \in [0, 1]^{784 \times |B|}$$
    /// @return the last layers of activations or the predictions, $$\hat{y} \in [0, 1]^{10 \times |B|}$$
    Matrix<double> forward(const Matrix<double> &input);

    /// @brief Updates the weights and biases of the network
    /// @param input several input datums in a single batch, $$x \in [0, 1]^{784 \times |B|}$$
    /// @param target true values, $$y \in \{0, 1\}^{10 \times |B|}$$
    void train(const Matrix<double> &input, const Matrix<double> &target);
};

} // namespace Batched
