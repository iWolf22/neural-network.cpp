#include "simple/neural_network.h"
#include "utils/images.h"
#include "utils/matrix.h"
#include "utils/utilities.h"

#include <gtest/gtest.h>
#include <stdexcept>

using Simple::NeuralNetwork;

namespace {

Matrix<uint8_t> make_image(uint8_t fill = 128) {
    Matrix<uint8_t> image(28, 28);
    for (size_t r = 0; r < image.rows(); ++r)
        for (size_t c = 0; c < image.cols(); ++c)
            image(r, c) = fill;
    return image;
}

void expect_vector_size(const Vector &v, size_t expected) {
    EXPECT_EQ(v.size(), expected);
}

} // namespace

TEST(SimpleNeuralNetworkConstructionTest, RejectsFewerThanTwoLayerSizes) {
    EXPECT_THROW(NeuralNetwork({784}), std::runtime_error);
    EXPECT_THROW(NeuralNetwork({}), std::runtime_error);
}

TEST(SimpleNeuralNetworkConstructionTest, AcceptsMinimalTwoLayerNetwork) {
    EXPECT_NO_THROW(NeuralNetwork({4, 2}));
}

TEST(SimpleNeuralNetworkForwardTest, OutputHasTenNeuronsForMnistArchitecture) {
    NeuralNetwork nn({784, 10});
    const Vector output = nn.forward(make_image());

    expect_vector_size(output, 10);
}

TEST(SimpleNeuralNetworkForwardTest, DeepNetworkProducesValidSigmoidOutputs) {
    NeuralNetwork nn({784, 16, 10});
    const Vector output = nn.forward(make_image(200));

    expect_vector_size(output, 10);
    for (double value : output) {
        EXPECT_GE(value, 0.0);
        EXPECT_LE(value, 1.0);
    }
}

TEST(SimpleNeuralNetworkForwardTest, SameImageYieldsDeterministicOutput) {
    NeuralNetwork nn({784, 8, 10});
    const Matrix<uint8_t> image = make_image(42);

    const Vector first = nn.forward(image);
    const Vector second = nn.forward(image);

    ASSERT_EQ(first.size(), second.size());
    for (size_t i = 0; i < first.size(); ++i)
        EXPECT_DOUBLE_EQ(first[i], second[i]);
}

TEST(SimpleNeuralNetworkTrainTest, SingleStepDoesNotThrow) {
    NeuralNetwork nn({784, 10});

    EXPECT_NO_THROW(nn.train(make_image(), 3, 0.01));
}

TEST(SimpleNeuralNetworkTrainTest, MultipleStepsChangeOutput) {
    NeuralNetwork nn({784, 10});
    const Matrix<uint8_t> image = make_image(100);

    const Vector before = nn.forward(image);

    for (int step = 0; step < 5; ++step)
        nn.train(image, 7, 0.1);

    const Vector after = nn.forward(image);

    bool changed = false;
    for (size_t i = 0; i < before.size(); ++i) {
        if (before[i] != after[i]) {
            changed = true;
            break;
        }
    }

    EXPECT_TRUE(changed);
}

TEST(SimpleNeuralNetworkIntegrationTest, ForwardWithRealMnistImage) {
    Images images(ImageType::TESTING);
    NeuralNetwork nn({784, 16, 10});

    const auto &sample = *images.begin();
    const Vector output = nn.forward(sample.image);

    expect_vector_size(output, 10);
    EXPECT_LE(sample.label, 9u);
}

TEST(SimpleNeuralNetworkTrainTest, AcceptsAllDigitLabels) {
    NeuralNetwork nn({784, 10});
    const Matrix<uint8_t> image = make_image(50);

    for (uint8_t label = 0; label < 10; ++label)
        EXPECT_NO_THROW(nn.train(image, label, 0.01));
}

TEST(SimpleNeuralNetworkForwardTest, SmallToyNetworkMatchesLayerSizes) {
    NeuralNetwork nn({3, 2, 1});
    Matrix<uint8_t> image(1, 3);
    image(0, 0) = 0;
    image(0, 1) = 128;
    image(0, 2) = 255;

    const Vector output = nn.forward(image);

    expect_vector_size(output, 1);
    EXPECT_GE(output[0], 0.0);
    EXPECT_LE(output[0], 1.0);
}

TEST(SimpleNeuralNetworkTrainTest, TrainingWithDifferentLabelsDoesNotThrow) {
    NeuralNetwork nn({784, 8, 10});
    const Matrix<uint8_t> image_a = make_image(10);
    const Matrix<uint8_t> image_b = make_image(240);

    EXPECT_NO_THROW({
        nn.train(image_a, 1, 0.05);
        nn.train(image_b, 8, 0.05);
    });
}
