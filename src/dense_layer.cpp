#include "dense_layer.h"
#include "utilities.h"

#include <stdexcept>

using std::runtime_error;

DenseLayer::DenseLayer(size_t input_size, size_t output_size)
    : weights_(output_size, input_size), biases_(output_size), input_cache_(input_size), z_cache_(output_size) {
    randomize_weights();
    randomize_biases();
}

inline void DenseLayer::randomize_weights() {
    for (size_t i = 0; i < weights_.rows(); ++i) {
        for (size_t j = 0; j < weights_.cols(); ++j) {
            weights_(i, j) = random_value();
        }
    }
}

inline void DenseLayer::randomize_biases() {
    for (size_t i = 0; i < biases_.size(); ++i) {
        biases_[i] = random_value();
    }
}

Vector DenseLayer::forward(const Vector &input) {
    if (input.size() != weights_.cols())
        throw runtime_error("Miss match in neurons");

    input_cache_ = input;

    // $$\textbf{z}^{(L)} = \textbf{W}\textbf{a}^{(L-1)} + \textbf{b}^{(L)}$$
    z_cache_ = (weights_ * input) + biases_;

    // $$\sigma(\textbf{z}^{(L)})$$
    return sigmoid(z_cache_);
}

Vector DenseLayer::backward(const Vector &grad_output, double learning_rate) {
    if (grad_output.size() != weights_.rows())
        throw runtime_error("Miss match in neurons");

    Vector grad_z = compute_grad_z(grad_output);
    update_weights(grad_z, learning_rate);
    update_biases(grad_z, learning_rate);
    return compute_grad_input(grad_z);
}

inline Vector DenseLayer::compute_grad_z(const Vector &grad_output) const {
    Vector grad_z(grad_output.size());

    for (size_t i = 0; i < grad_output.size(); ++i)
        grad_z[i] = grad_output[i] * sigmoid_derivative(z_cache_[i]);

    return grad_z;
}

inline Vector DenseLayer::compute_grad_input(const Vector &grad_z) const {
    Vector grad_input(input_cache_.size());

    for (size_t i = 0; i < weights_.rows(); ++i)
        for (size_t j = 0; j < weights_.cols(); ++j)
            grad_input[j] += grad_z[i] * weights_(i, j);

    return grad_input;
}

inline void DenseLayer::update_weights(const Vector &grad_z, double learning_rate) {
    for (size_t i = 0; i < weights_.rows(); ++i)
        for (size_t j = 0; j < weights_.cols(); ++j)
            weights_(i, j) -= learning_rate * grad_z[i] * input_cache_[j];
}

inline void DenseLayer::update_biases(const Vector &grad_z, double learning_rate) {
    for (size_t i = 0; i < weights_.rows(); ++i)
        biases_[i] -= learning_rate * grad_z[i];
}
