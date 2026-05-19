#include "simple/neural_network.h"
#include "utils/utilities.h"

#include <stdexcept>

using std::initializer_list;
using std::runtime_error;

NeuralNetwork::NeuralNetwork(initializer_list<unsigned int> layer_sizes) {
    if (layer_sizes.size() < 2)
        throw runtime_error("Fewer than 2 layers in the neural network");

    layers_.reserve(layer_sizes.size() - 1);

    auto a = layer_sizes.begin();
    auto b = std::next(a);

    while (b != layer_sizes.end()) {
        layers_.emplace_back(*a, *b);
        ++a;
        ++b;
    }
}

Vector NeuralNetwork::forward(const Matrix<uint8_t> &input) {
    auto it = layers_.begin();

    Vector cur = it->forward(matrix_to_vector(input));
    ++it;

    for (; it != layers_.end(); ++it)
        cur = it->forward(cur);

    return cur;
}

Vector compute_grad_loss(const Vector &output, const Vector &target) {
    Vector grad_loss(output.size());

    for (size_t i = 0; i < output.size(); ++i)
        grad_loss[i] = 2.0 * (output[i] - target[i]);

    return grad_loss;
}

void NeuralNetwork::train(const Matrix<uint8_t> &input, uint8_t target, double learning_rate) {
    Vector output = forward(input);
    Vector grad_loss = compute_grad_loss(output, label_to_vector(target));
    backward(grad_loss, learning_rate);
}

void NeuralNetwork::backward(const Vector &grad_lost, double learning_rate) {
    auto it = layers_.rbegin();

    Vector cur = it->backward(grad_lost, learning_rate);
    ++it;

    for (; it != layers_.rend(); ++it)
        cur = it->backward(cur, learning_rate);
}
