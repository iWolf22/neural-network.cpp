#include "images.h"
#include "neural_network.h"

#include <iostream>

double LEARNING_RATE = 0.001;

int main(int argc, char *argv[]) {
    // Printing out the digits
    // Images image = Images(ImageType::TRAINING);
    // std::cout << image << std::endl;

    NeuralNetwork nn{784, 84, 84, 10};

    for (int a = 0; a < 10; ++a) {
        Images training_images(ImageType::TRAINING);

        int i = 0;

        for (const auto &labelled_image : training_images) {
            std::cout << a << " - " << i << std::endl;
            nn.train(labelled_image.image, labelled_image.label, LEARNING_RATE);
            ++i;
        }
    }

    Images testing_images(ImageType::TESTING);
    int j = 0;

    for (const auto &labelled_image : testing_images) {
        Vector result = nn.forward(labelled_image.image);

        std::cout << j << std::endl;
        std::cout << "True Value: " << +labelled_image.label << "\n";
        for (const auto &val : result)
            std::cout << val << " ";
        std::cout << "\n\n";

        ++j;
        if (j > 20)
            break;
    }
}
