#include "batched/neural_network.h"
#include "utils/utilities.h"

#include <stdexcept>

using std::initializer_list;
using std::runtime_error;
using std::vector;

namespace Batched {

NeuralNetwork::NeuralNetwork(double learning_rate, size_t batch_size, initializer_list<unsigned int> layer_sizes)
    : learning_rate_{learning_rate}, batch_size_{batch_size} {
    if (layer_sizes.size() < 2)
        throw runtime_error("Fewer than 2 layers in the neural network");

    layers_.reserve(layer_sizes.size() - 1);

    auto a = layer_sizes.begin();
    auto b = std::next(a);

    for (; b != layer_sizes.end(); ++a, ++b)
        layers_.emplace_back(learning_rate_, batch_size_, *a, *b);
}

Matrix<double> NeuralNetwork::forward(const Matrix<double> &input) {
    auto it = layers_.begin();

    Matrix<double> cur = it->forward(input);
    ++it;

    for (; it != layers_.end(); ++it)
        cur = it->forward(cur);

    return cur;
}

void NeuralNetwork::train(const Matrix<double> &input, const Matrix<double> &target) {
    Matrix<double> output = forward(input);
    Matrix<double> grad_loss = 2.0 * (output + (-1.0 * target));
    backward(grad_loss);
}

void NeuralNetwork::backward(const Matrix<double> &grad_lost) {
    auto it = layers_.rbegin();

    Matrix<double> cur = it->backward(grad_lost);
    ++it;

    for (; it != layers_.rend(); ++it)
        cur = it->backward(cur);
}

} // namespace Batched
