#include "simple/dense_layer.h"
#include "utils/aliases.h"
#include "utils/utilities.h"

#include <gtest/gtest.h>
#include <stdexcept>

using Simple::DenseLayer;

namespace {

Vector make_input(size_t size, double fill = 0.5) {
    Vector v(size);
    for (size_t i = 0; i < size; ++i)
        v[i] = fill;
    return v;
}

Vector make_grad(size_t size) {
    Vector v(size);
    for (size_t i = 0; i < size; ++i)
        v[i] = 0.1 * static_cast<double>(i + 1);
    return v;
}

} // namespace

TEST(SimpleDenseLayerConstructionTest, CanCreateLayerWithMatchingSizes) {
    DenseLayer layer(4, 2);
    const Vector input = make_input(4);
    const Vector output = layer.forward(input);

    EXPECT_EQ(output.size(), 2u);
}

TEST(SimpleDenseLayerForwardTest, OutputValuesAreInUnitInterval) {
    DenseLayer layer(3, 3);
    const Vector output = layer.forward(make_input(3));

    for (double value : output) {
        EXPECT_GE(value, 0.0);
        EXPECT_LE(value, 1.0);
    }
}

TEST(SimpleDenseLayerForwardTest, SameInputProducesSameOutput) {
    DenseLayer layer(5, 2);
    const Vector input = make_input(5, 0.25);

    const Vector first = layer.forward(input);
    const Vector second = layer.forward(input);

    ASSERT_EQ(first.size(), second.size());
    for (size_t i = 0; i < first.size(); ++i)
        EXPECT_DOUBLE_EQ(first[i], second[i]);
}

TEST(SimpleDenseLayerForwardTest, WrongInputSizeThrows) {
    DenseLayer layer(4, 2);

    EXPECT_THROW(layer.forward(make_input(3)), std::runtime_error);
    EXPECT_THROW(layer.forward(make_input(5)), std::runtime_error);
}

TEST(SimpleDenseLayerBackwardTest, ReturnsGradientForPreviousLayer) {
    DenseLayer layer(4, 3);
    layer.forward(make_input(4));

    const Vector grad_input = layer.backward(make_grad(3), 0.01);

    EXPECT_EQ(grad_input.size(), 4u);
}

TEST(SimpleDenseLayerBackwardTest, WrongGradientSizeThrows) {
    DenseLayer layer(4, 2);
    layer.forward(make_input(4));

    EXPECT_THROW(layer.backward(make_grad(3), 0.01), std::runtime_error);
    EXPECT_THROW(layer.backward(make_grad(1), 0.01), std::runtime_error);
}

TEST(SimpleDenseLayerBackwardTest, ForwardBackwardRoundTripDoesNotThrow) {
    DenseLayer layer(6, 4);
    const Vector input = make_input(6, 0.3);

    EXPECT_NO_THROW({
        const Vector output = layer.forward(input);
        const Vector grad_input = layer.backward(make_grad(output.size()), 0.05);
        EXPECT_EQ(grad_input.size(), input.size());
    });
}

TEST(SimpleDenseLayerIntegrationTest, ChainedLayersPropagateActivations) {
    DenseLayer hidden(4, 3);
    DenseLayer output(3, 2);

    const Vector hidden_out = hidden.forward(make_input(4));
    const Vector final_out = output.forward(hidden_out);

    EXPECT_EQ(final_out.size(), 2u);
    for (double value : final_out) {
        EXPECT_GE(value, 0.0);
        EXPECT_LE(value, 1.0);
    }
}

TEST(SimpleDenseLayerIntegrationTest, BackwardThroughTwoLayersReturnsInputSizedGradient) {
    DenseLayer hidden(3, 2);
    DenseLayer output(2, 1);

    const Vector hidden_out = hidden.forward(make_input(3));
    const Vector final_out = output.forward(hidden_out);

    output.backward(make_grad(final_out.size()), 0.02);
    const Vector grad_input = hidden.backward(make_grad(2), 0.02);

    EXPECT_EQ(grad_input.size(), 3u);
}

TEST(SimpleDenseLayerSizingTest, SupportsWideLayers) {
    DenseLayer layer(1, 8);
    const Vector output = layer.forward(make_input(1));

    EXPECT_EQ(output.size(), 8u);
}

TEST(SimpleDenseLayerSizingTest, SupportsDeepInputVectors) {
    DenseLayer layer(32, 4);
    const Vector output = layer.forward(make_input(32, 0.1));

    EXPECT_EQ(output.size(), 4u);
}
