#pragma once

#include "batched/neural_network.h"

void train(Batched::NeuralNetwork &nn, int epochs);

void inference(Batched::NeuralNetwork &nn);
