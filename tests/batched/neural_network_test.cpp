#include "batched/neural_network.h"
#include "utils/matrix.h"
#include "utils/utilities.h"

#include <gtest/gtest.h>
#include <stdexcept>

using Batched::NeuralNetwork;

namespace {

Matrix<double> make_batch(size_t rows, size_t cols, double fill = 0.5) {
    Matrix<double> m(rows, cols);
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c)
            m(r, c) = fill + static_cast<double>(r) * 0.01 + static_cast<double>(c) * 0.001;
    return m;
}

Matrix<double> make_targets(size_t rows, size_t cols, size_t active_row = 0) {
    Matrix<double> targets(rows, cols);
    for (size_t c = 0; c < cols; ++c)
        for (size_t r = 0; r < rows; ++r)
            targets(r, c) = (r == active_row) ? 1.0 : 0.0;
    return targets;
}

} // namespace

TEST(BatchedNeuralNetworkConstructionTest, RejectsFewerThanTwoLayerSizes) {
    EXPECT_THROW(NeuralNetwork(0.1, 4, {784}), std::runtime_error);
}

TEST(BatchedNeuralNetworkConstructionTest, StoresLearningRateAndBatchSize) {
    const NeuralNetwork nn(0.05, 8, {4, 3, 2});

    EXPECT_DOUBLE_EQ(nn.learning_rate_, 0.05);
    EXPECT_EQ(nn.batch_size_, 8u);
}

TEST(BatchedNeuralNetworkForwardTest, OutputColumnsMatchBatchSize) {
    constexpr size_t batch = 3;
    NeuralNetwork nn(0.1, batch, {4, 3, 2});

    const Matrix<double> output = nn.forward(make_batch(4, batch));

    EXPECT_EQ(output.rows(), 2u);
    EXPECT_EQ(output.cols(), batch);
}

TEST(BatchedNeuralNetworkForwardTest, ActivationsStayWithinSigmoidRange) {
    NeuralNetwork nn(0.1, 2, {6, 4, 2});
    const Matrix<double> output = nn.forward(make_batch(6, 2, 0.2));

    for (size_t r = 0; r < output.rows(); ++r)
        for (size_t c = 0; c < output.cols(); ++c) {
            EXPECT_GE(output(r, c), 0.0) << "row " << r << " col " << c;
            EXPECT_LE(output(r, c), 1.0) << "row " << r << " col " << c;
        }
}

TEST(BatchedNeuralNetworkForwardTest, SameBatchIsDeterministic) {
    NeuralNetwork nn(0.1, 2, {5, 3});
    const Matrix<double> input = make_batch(5, 2, 0.7);

    const Matrix<double> first = nn.forward(input);
    const Matrix<double> second = nn.forward(input);

    ASSERT_EQ(first.rows(), second.rows());
    ASSERT_EQ(first.cols(), second.cols());
    for (size_t r = 0; r < first.rows(); ++r)
        for (size_t c = 0; c < first.cols(); ++c)
            EXPECT_DOUBLE_EQ(first(r, c), second(r, c));
}

TEST(BatchedNeuralNetworkTrainTest, SingleBatchUpdateDoesNotThrow) {
    NeuralNetwork nn(0.1, 2, {4, 3, 2});
    const Matrix<double> input = make_batch(4, 2);
    const Matrix<double> targets = make_targets(2, 2, 1);

    EXPECT_NO_THROW(nn.train(input, targets));
}

TEST(BatchedNeuralNetworkTrainTest, TrainingChangesForwardOutput) {
    NeuralNetwork nn(0.5, 2, {4, 3, 2});
    const Matrix<double> input = make_batch(4, 2, 0.4);
    const Matrix<double> targets = make_targets(2, 2, 0);

    const Matrix<double> before = nn.forward(input);

    for (int step = 0; step < 10; ++step)
        nn.train(input, targets);

    const Matrix<double> after = nn.forward(input);

    bool changed = false;
    for (size_t r = 0; r < before.rows(); ++r)
        for (size_t c = 0; c < before.cols(); ++c) {
            if (before(r, c) != after(r, c)) {
                changed = true;
                break;
            }
        }

    EXPECT_TRUE(changed);
}

TEST(BatchedNeuralNetworkIntegrationTest, MnistSizedArchitectureForward) {
    constexpr size_t batch = 4;
    NeuralNetwork nn(0.01, batch, {784, 16, 10});

    const Matrix<double> input = make_batch(784, batch, 0.1);
    const Matrix<double> output = nn.forward(input);

    EXPECT_EQ(output.rows(), 10u);
    EXPECT_EQ(output.cols(), batch);
}

TEST(BatchedNeuralNetworkTrainTest, HandlesDifferentTargetColumns) {
    NeuralNetwork nn(0.1, 3, {4, 2});
    const Matrix<double> input = make_batch(4, 3);
    Matrix<double> targets(2, 3);

    targets(0, 0) = 1;
    targets(1, 1) = 1;
    targets(0, 2) = 1;

    EXPECT_NO_THROW(nn.train(input, targets));
}

TEST(BatchedNeuralNetworkForwardTest, SingleColumnBatchWorks) {
    NeuralNetwork nn(0.1, 1, {3, 2});
    const Matrix<double> output = nn.forward(make_batch(3, 1));

    EXPECT_EQ(output.rows(), 2u);
    EXPECT_EQ(output.cols(), 1u);
}

TEST(BatchedNeuralNetworkTrainTest, RepeatedTrainingDoesNotThrow) {
    NeuralNetwork nn(0.2, 2, {5, 4, 3});
    const Matrix<double> input = make_batch(5, 2);
    const Matrix<double> targets = make_targets(3, 2, 2);

    for (int epoch = 0; epoch < 3; ++epoch)
        EXPECT_NO_THROW(nn.train(input, targets));
}
