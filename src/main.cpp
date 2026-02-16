#include "images.h"

#include <iostream>

int main(int argc, char *argv[]) {
    Images image = Images(ImageType::TRAINING);
    std::cout << image << std::endl;
}
