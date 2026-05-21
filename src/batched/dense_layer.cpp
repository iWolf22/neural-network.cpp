#include "batched/dense_layer.h"
#include "utils/aliases.h"
#include "utils/utilities.h"

#include <stdexcept>

using std::runtime_error;

namespace Batched {

DenseLayer::DenseLayer(double learning_rate, size_t batch_size, size_t input_size, size_t output_size)
    : learning_rate_{learning_rate}, batch_size_{batch_size}, weights_(output_size, input_size), biases_(output_size),
      input_cache_(input_size, batch_size), output_cache_(output_size, batch_size) {
    randomize_weights();
    randomize_biases();
}

inline void DenseLayer::randomize_weights() {
    for (size_t i = 0; i < weights_.rows(); ++i)
        for (size_t j = 0; j < weights_.cols(); ++j)
            weights_(i, j) = random_value();
}

inline void DenseLayer::randomize_biases() {
    for (size_t i = 0; i < biases_.size(); ++i)
        biases_[i] = random_value();
}

const Matrix<double> DenseLayer::forward(const Matrix<double> &input) {
    input_cache_ = input;

    output_cache_ = sigmoid((weights_ * input) + biases_);

    return output_cache_;
}

Vector bias_sum(Matrix<double> &B) {
    Vector result(B.rows());

    for (size_t i = 0; i < B.rows(); ++i)
        for (size_t j = 0; j < B.cols(); ++j)
            result[i] += B(i, j);

    return result;
}

const Matrix<double> DenseLayer::backward(const Matrix<double> &grad_output) {
    Matrix<double> delta = hadamard(grad_output, sigmoid_derivative(output_cache_));

    Matrix<double> grad_input = transpose(weights_) * delta;

    weights_ = weights_ + (-learning_rate_ / batch_size_) * (delta * transpose(input_cache_));

    biases_ = biases_ + (-learning_rate_ / batch_size_) * bias_sum(delta);

    return grad_input;
}

} // namespace Batched