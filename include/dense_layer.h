#pragma once

#include "aliases.h"
#include "matrix.h"

/// @brief One layer in the neural network, $$^{(L)}$$ for
/// $$L \in \{1, 2, 3, 4\}$$
class DenseLayer {
    /// @brief weights_(i, j) = $$w_{i,j}^{(L)}$$
    Matrix<double> weights_;

    /// @brief biases_[i] = $$b_i^{(L)}$$
    Vector biases_;

    /// @brief input_cache_[i] = $$a_i^{(L-1)}$$
    Vector input_cache_;

    /// @brief z_cache_[i] = $$z_i^{(L)}$$ (known as the pre-activation or the weighted sum)
    Vector z_cache_;

    inline void randomize_weights();
    inline void randomize_biases();

    /// @return grad_z[i] = $$\tfrac{\partial C_x}{\partial z_i^{(L)}}$$
    inline Vector compute_grad_z(const Vector &grad_output) const;

    /// @return grad_input[i] = $$\tfrac{\partial C_x}{\partial a_i^{(L-1)}}$$
    inline Vector compute_grad_input(const Vector &grad_z) const;

    inline void update_weights(const Vector &grad_z, double learning_rate);
    inline void update_biases(const Vector &grad_z, double learning_rate);

  public:
    /// @param input_size number of neurons in the previous layer, $$^{(L-1)}$$
    /// @param output_size number of neurons in the current layer, $$^{(L)}$$
    DenseLayer(size_t input_size, size_t output_size);

    /// @param input input[i] = $$a_i^{(L-1)}$$
    /// @return next layer of activation values, $$a_i^{(L)}$$
    Vector forward(const Vector &input);

    /// @param grad_output grad_output[i] = $$\tfrac{\partial C_x}{\partial a_i^{(L)}}$$
    /// @param learning_rate gradient descent step size
    /// @return grad_input[i] = $$\tfrac{\partial C_x}{\partial a_i^{(L-1)}}$$
    Vector backward(const Vector &grad_output, double learning_rate);
};
