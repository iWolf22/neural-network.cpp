#include "batched/dense_layer.h"
#include "utils/matrix.h"
#include "utils/utilities.h"

#include <gtest/gtest.h>

using Batched::DenseLayer;

namespace {

Matrix<double> make_batch(size_t rows, size_t cols, double fill = 0.5) {
    Matrix<double> m(rows, cols);
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c)
            m(r, c) = fill + static_cast<double>(r) * 0.01 + static_cast<double>(c) * 0.001;
    return m;
}

Matrix<double> make_grad(size_t rows, size_t cols) {
    Matrix<double> m(rows, cols);
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c)
            m(r, c) = 0.05 * static_cast<double>(r + c + 1);
    return m;
}

} // namespace

TEST(BatchedDenseLayerForwardTest, OutputShapeMatchesLayerSizes) {
    constexpr size_t batch = 3;
    DenseLayer layer(0.1, batch, 4, 2);

    const Matrix<double> output = layer.forward(make_batch(4, batch));

    EXPECT_EQ(output.rows(), 2u);
    EXPECT_EQ(output.cols(), batch);
}

TEST(BatchedDenseLayerForwardTest, ActivationsAreSigmoidBounded) {
    constexpr size_t batch = 2;
    DenseLayer layer(0.1, batch, 5, 3);
    const Matrix<double> output = layer.forward(make_batch(5, batch, 0.2));

    for (size_t r = 0; r < output.rows(); ++r)
        for (size_t c = 0; c < output.cols(); ++c) {
            EXPECT_GE(output(r, c), 0.0);
            EXPECT_LE(output(r, c), 1.0);
        }
}

TEST(BatchedDenseLayerForwardTest, SameInputIsDeterministic) {
    constexpr size_t batch = 2;
    DenseLayer layer(0.1, batch, 3, 2);
    const Matrix<double> input = make_batch(3, batch, 0.7);

    const Matrix<double> first = layer.forward(input);
    const Matrix<double> second = layer.forward(input);

    for (size_t r = 0; r < first.rows(); ++r)
        for (size_t c = 0; c < first.cols(); ++c)
            EXPECT_DOUBLE_EQ(first(r, c), second(r, c));
}

TEST(BatchedDenseLayerBackwardTest, ReturnsInputSizedGradient) {
    constexpr size_t batch = 2;
    DenseLayer layer(0.1, batch, 4, 3);

    layer.forward(make_batch(4, batch));
    const Matrix<double> grad_input = layer.backward(make_grad(3, batch));

    EXPECT_EQ(grad_input.rows(), 4u);
    EXPECT_EQ(grad_input.cols(), batch);
}

TEST(BatchedDenseLayerBackwardTest, UpdatesWeightsSoForwardOutputChanges) {
    constexpr size_t batch = 2;
    DenseLayer layer(0.5, batch, 3, 2);
    const Matrix<double> input = make_batch(3, batch, 0.4);

    const Matrix<double> before = layer.forward(input);
    layer.backward(make_grad(2, batch));
    const Matrix<double> after = layer.forward(input);

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

TEST(BatchedDenseLayerBackwardTest, ForwardBackwardRoundTripDoesNotThrow) {
    constexpr size_t batch = 3;
    DenseLayer layer(0.1, batch, 6, 4);
    const Matrix<double> input = make_batch(6, batch);

    EXPECT_NO_THROW({
        const Matrix<double> output = layer.forward(input);
        const Matrix<double> grad_input = layer.backward(make_grad(output.rows(), output.cols()));
        EXPECT_EQ(grad_input.rows(), input.rows());
        EXPECT_EQ(grad_input.cols(), input.cols());
    });
}

TEST(BatchedDenseLayerIntegrationTest, ChainedLayersPropagateBatch) {
    constexpr size_t batch = 2;
    DenseLayer hidden(0.1, batch, 4, 3);
    DenseLayer output(0.1, batch, 3, 2);

    const Matrix<double> hidden_out = hidden.forward(make_batch(4, batch));
    const Matrix<double> final_out = output.forward(hidden_out);

    EXPECT_EQ(final_out.rows(), 2u);
    EXPECT_EQ(final_out.cols(), batch);
}

TEST(BatchedDenseLayerIntegrationTest, BackwardThroughTwoLayersMatchesInput) {
    constexpr size_t batch = 2;
    DenseLayer hidden(0.2, batch, 3, 2);
    DenseLayer output(0.2, batch, 2, 1);

    const Matrix<double> input = make_batch(3, batch);
    const Matrix<double> hidden_out = hidden.forward(input);
    const Matrix<double> final_out = output.forward(hidden_out);

    output.backward(make_grad(final_out.rows(), batch));
    const Matrix<double> grad_input = hidden.backward(make_grad(2, batch));

    EXPECT_EQ(grad_input.rows(), input.rows());
    EXPECT_EQ(grad_input.cols(), input.cols());
}

TEST(BatchedDenseLayerSizingTest, SingleColumnBatchWorks) {
    DenseLayer layer(0.1, 1, 3, 2);
    const Matrix<double> output = layer.forward(make_batch(3, 1));

    EXPECT_EQ(output.rows(), 2u);
    EXPECT_EQ(output.cols(), 1u);
}

TEST(BatchedDenseLayerSizingTest, WideOutputLayer) {
    constexpr size_t batch = 4;
    DenseLayer layer(0.01, batch, 2, 8);
    const Matrix<double> output = layer.forward(make_batch(2, batch));

    EXPECT_EQ(output.rows(), 8u);
    EXPECT_EQ(output.cols(), batch);
}
