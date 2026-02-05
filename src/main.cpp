#include "utils/images.h"

#include <iostream>

int main(int argc, char *argv[]) {
    Images image = Images(ImageType::TESTING);
    std::cout << image << std::endl;
}
