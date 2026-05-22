#pragma once

#include "utils/aliases.h"
#include "utils/matrix.h"

namespace Batched {

/// @brief One layer in the neural network, $$^{(L)}$$ for $$L \in \{1, 2, 3, 4\}$$
class DenseLayer {
    double learning_rate_;
    size_t batch_size_;

    /// @brief weights_(i, j) = $$w_{i,j}^{(L)}$$
    Matrix<double> weights_;

    /// @brief biases_[i] = $$b_i^{(L)}$$
    Vector biases_;

    /// @brief $$\textbf{A}^{(L-1)}$$
    Matrix<double> input_cache_;

    /// @brief $$\textbf{A}^{(L)}$$
    Matrix<double> output_cache_;

    inline void randomize_weights();
    inline void randomize_biases();

  public:
    /// @param learning_rate gradient descent step size
    /// @param input_size number of neurons in the previous layer, $$^{(L-1)}$$
    /// @param output_size number of neurons in the current layer, $$^{(L)}$$
    DenseLayer(double learning_rate, size_t batch_size, size_t input_size, size_t output_size);

    /// @param input $$\textbf{A}^{(L-1)}$$
    /// @return $$\textbf{A}^{(L)} = \sigma(\textbf{W}^{(L)}\textbf{A}^{(L-1)} + \textbf{B}^{(L)})$$
    const Matrix<double> forward(const Matrix<double> &input);

    /// @param grad_output $$\tfrac{\partial C_x}{\partial \textbf{A}^{(L)}}$$
    /// @return $$\tfrac{\partial C_x}{\partial \textbf{A}^{(L-1)}}$$
    const Matrix<double> backward(const Matrix<double> &grad_output);
};

} // namespace Batched
