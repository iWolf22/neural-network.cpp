#pragma once

#include "aliases.h"
#include "matrix.h"

/// @brief One layer in the neural network, $$^{(L)}$$ for
/// $$L \in \{1, 2, 3, 4\}$$
class DenseLayer {
    /// @brief weights(i, j) = $$w_{i,j}^{(L)}$$
    Matrix<double> weights;

    /// @brief biases[i] = $$b_i^{(L)}$$
    Vector biases;

    /// @brief inputCache[i] = $$a_i^{(L-1)}$$
    Vector inputCache;

    /// @brief zCache[i] = $$z_i^{(L)}$$ (known as the pre-activation or the weighted sum)
    Vector zCache;

  public:
    /// @param inputSize number of neurons in the previous layer, $$^{(L-1)}$$
    /// @param outputSize number of neurons in the current layer, $$^{(L)}$$
    DenseLayer(size_t inputSize, size_t outputSize);

    /// @param input input[i] = $$a_i^{(L-1)}$$
    /// @return next layer of activation values, $$a_i^{(L)}$$
    Vector forward(const Vector &input);

    /// @param gradOutput gradOutput[i] = $$\tfrac{\partial C_x}{\partial a_i^{(L)}}$$
    /// @param learningRate gradient descent step size
    /// @return gradInput[i] = $$\tfrac{\partial C_x}{\partial a_i^{(L-1)}}$$
    Vector backward(const Vector &gradOutput, double learningRate);
};
