#include "utils/helper.h"
#include "utils/images.h"
#include "utils/matrix.h"

#include <iostream>

using std::cout;
using std::vector;

void fill_train_column(Matrix<double> &train, size_t col, const Matrix<uint8_t> &image) {
    const size_t rows = image.rows();
    const size_t cols = image.cols();

    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            train(i * cols + j, col) = static_cast<double>(image(i, j)) / 255.0;
}

void fill_label_column(Matrix<double> &results, size_t col, uint8_t label) {
    for (size_t i = 0; i < 10; ++i)
        results(i, col) = (i == label) ? 1 : 0;
}

void train(Batched::NeuralNetwork &nn, int epochs) {
    Images images(ImageType::TRAINING);

    Matrix<double> train(784, nn.batch_size_);
    Matrix<double> results(10, nn.batch_size_);

    size_t n = 0;
    int c = 1;

    for (int e = 0; e < epochs; ++e) {
        for (const auto &labelled_image : images) {
            fill_train_column(train, n, labelled_image.image);
            fill_label_column(results, n, labelled_image.label);

            ++n;

            if (n == nn.batch_size_) {
                cout << "training... " << e << " - " << c * nn.batch_size_ << "\n";
                nn.train(train, results);
                n = 0;
                ++c;
            }
        }
    }
}

vector<uint8_t> arg_max(const Matrix<double> &result) {
    vector<uint8_t> res;
    res.reserve(result.cols());

    for (size_t col = 0; col < result.cols(); ++col) {
        uint8_t best_row = 0;
        double best_val = result(0, col);

        for (uint8_t row = 1; row < result.rows(); ++row) {
            if (result(row, col) > best_val) {
                best_val = result(row, col);
                best_row = row;
            }
        }

        res.push_back(best_row);
    }

    return res;
}

void output_confusion_matrix(const Matrix<int> &confusion_matrix) {
    cout << "Confusion matrix:\n";
    for (size_t i = 0; i < confusion_matrix.rows(); ++i) {
        for (size_t j = 0; j < confusion_matrix.cols(); ++j)
            cout << confusion_matrix(i, j) << " ";
        cout << "\n";
    }
}

void process_result(const vector<uint8_t> &result, const vector<Images::LabelledImage> batch,
                    Matrix<int> &confusion_matrix) {
    for (size_t i = 0; i < result.size(); ++i) {
        confusion_matrix(result[i], batch[i].label) += 1;
        if (result[i] != batch[i].label) {
            cout << "Prediction: " << static_cast<int>(result[i]) << "\n";
            cout << batch[i];
        }
    }
}

void inference(Batched::NeuralNetwork &nn) {
    Images images(ImageType::TESTING);

    Matrix<double> test(784, nn.batch_size_);
    vector<Images::LabelledImage> batch;
    batch.reserve(nn.batch_size_);
    Matrix<int> confusion_matrix(10, 10);

    size_t n = 0;
    int c = 1;

    for (const auto &labelled_image : images) {
        fill_train_column(test, n, labelled_image.image);
        batch.push_back(labelled_image);

        ++n;

        if (n == nn.batch_size_) {
            cout << "testing... " << c * nn.batch_size_ << std::endl;
            process_result(arg_max(nn.forward(test)), batch, confusion_matrix);
            batch.clear();
            n = 0;
            ++c;
        }
    }

    output_confusion_matrix(confusion_matrix);
}
