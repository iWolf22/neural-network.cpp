#include "batched/neural_network.h"
#include "utils/images.h"

#include <iostream>

// params
double LEARNING_RATE = 0.01;
int BATCH_SIZE = 64;

int main(int argc, char *argv[]) {
    // Printing out the digits
    // Images image = Images(ImageType::TRAINING);
    // std::cout << image << std::endl;

    Batched::NeuralNetwork nn(LEARNING_RATE, BATCH_SIZE, {784, 128, 128, 10});

    Images training_images(ImageType::TRAINING);

    Matrix<double> train(784, BATCH_SIZE);
    Matrix<double> results(10, BATCH_SIZE);

    int n = 0;
    int c = 1;

    for (size_t k = 0; k < 10; ++k) {
        for (const auto &labelled_image : training_images) {
            for (size_t i = 0; i < labelled_image.image.rows(); ++i)
                for (size_t j = 0; j < labelled_image.image.cols(); ++j)
                    train(i * 28 + j, n) = static_cast<double>(labelled_image.image(i, j)) / 255.0;

            for (size_t i = 0; i < 10; ++i) {
                if (i == labelled_image.label)
                    results(i, n) = 1;
                else
                    results(i, n) = 0;
            }

            ++n;

            if (n == BATCH_SIZE) {
                std::cout << "train - " << k << " - " << c * BATCH_SIZE << std::endl;
                nn.train(train, results);
                n = 0;
                ++c;
            }
        }
    }

    for (const auto &labelled_image : training_images) {
        for (size_t i = 0; i < labelled_image.image.rows(); ++i)
            for (size_t j = 0; j < labelled_image.image.cols(); ++j)
                train(i * 28 + j, n) = static_cast<double>(labelled_image.image(i, j)) / 255.0;

        results(0, n) = labelled_image.label;

        ++n;

        if (n == BATCH_SIZE) {
            Matrix<double> res = nn.forward(train);

            for (size_t i = 0; i < res.cols(); ++i) {
                std::cout << results(0, i) << "\n";

                for (size_t j = 0; j < res.rows(); ++j) {
                    std::cout << res(j, i) << " ";
                }

                std::cout << "\n\n";
            }

            break;
        }
    }
}
