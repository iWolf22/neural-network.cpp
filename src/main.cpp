#include "batched/neural_network.h"
#include "utils/helper.h"

#include <iostream>

// Hyperparameters
double LEARNING_RATE = 0.01;
constexpr int BATCH_SIZE = 64;
int EPOCHS = 30;

int main(int argc, char *argv[]) {
    Batched::NeuralNetwork nn(LEARNING_RATE, BATCH_SIZE, {784, 128, 64, 10});
    train(nn, EPOCHS);
    inference(nn);
}
