#include "utils/images.h"

#include <bit>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using std::byteswap;
using std::ifstream;
using std::invalid_argument;
using std::istream;
using std::ostream;
using std::runtime_error;
using std::string;
using std::to_string;
using std::filesystem::path;

using const_iterator = Images::const_iterator;
using LabelledImage = Images::LabelledImage;

struct ImagesEndTag {};

constexpr uint32_t MNIST_IMAGE_MAGIC = 0x00000803;
constexpr uint32_t MNIST_LABEL_MAGIC = 0x00000801;

const int MNIST_IMAGE_HEADER_SIZE = 4 * 4;
const int MNIST_LABEL_HEADER_SIZE = 2 * 4;

uint32_t read_u32(istream &file) {
    uint32_t word = 0;
    file.read(reinterpret_cast<char *>(&word), 4);
    return word;
}

uint32_t read_u32_be(istream &file) {
    uint32_t word = read_u32(file);
    if constexpr (std::endian::native == std::endian::little) {
        return byteswap(word);
    }
    return word;
}

Images::Images(ImageType image_type, path data_dir) {
    get_mnist_filepaths(image_type, data_dir);

    ifstream images_file = open_file(FileType::IMAGES);
    ifstream labels_file = open_file(FileType::LABELS);

    const uint32_t images_magic = read_u32_be(images_file);
    if (images_magic != MNIST_IMAGE_MAGIC) {
        throw runtime_error("Invalid MNIST images magic number");
    }

    const uint32_t labels_magic = read_u32_be(labels_file);
    if (labels_magic != MNIST_LABEL_MAGIC) {
        throw runtime_error("Invalid MNIST labels magic number");
    }

    const uint32_t images_count = read_u32_be(images_file);
    const uint32_t labels_count = read_u32_be(labels_file);
    if (images_count != labels_count) {
        throw runtime_error("Mismatching MNIST file sizes");
    }

    size_ = images_count;
    rows_ = read_u32_be(images_file);
    cols_ = read_u32_be(images_file);
}

void Images::get_mnist_filepaths(ImageType image_type, const path &data_dir) {
    switch (image_type) {
    case ImageType::TRAINING:
        images_filename_ = data_dir / "train-images.idx3-ubyte";
        labels_filename_ = data_dir / "train-labels.idx1-ubyte";
        break;
    case ImageType::TESTING:
        images_filename_ = data_dir / "t10k-images.idx3-ubyte";
        labels_filename_ = data_dir / "t10k-labels.idx1-ubyte";
        break;
    default:
        throw invalid_argument("Invalid ImageType provided");
    }
}

ifstream Images::open_file(FileType file_type) const {
    ifstream file(file_type == FileType::IMAGES ? images_filename_ : labels_filename_, std::ios::binary);
    file.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    return file;
}

size_t Images::size() const { return size_; }

uint32_t Images::rows() const { return rows_; }

uint32_t Images::cols() const { return cols_; }

const_iterator Images::begin() const { return const_iterator(*this); }

const_iterator Images::end() const { return const_iterator(*this, ImagesEndTag{}); }

const_iterator::const_iterator(const Images &owner)
    : owner_{owner}, images_file_{owner.open_file(FileType::IMAGES)}, labels_file_{owner.open_file(FileType::LABELS)},
      labelled_image_{LabelledImage(Matrix<uint8_t>(owner.rows(), owner.cols()), 0)}, index_{} {
    skip_headers();
    read_image();
}

const_iterator::const_iterator(const Images &owner, ImagesEndTag)
    : owner_{owner}, labelled_image_{LabelledImage(Matrix<uint8_t>(0, 0), 0)}, index_{owner.size()} {}

void const_iterator::skip_headers() {
    images_file_.seekg(MNIST_IMAGE_HEADER_SIZE);
    labels_file_.seekg(MNIST_LABEL_HEADER_SIZE);
}

void const_iterator::read_image() {
    labels_file_.read(reinterpret_cast<char *>(&labelled_image_.label), 1);

    Matrix<uint8_t> &image = labelled_image_.image;

    for (size_t i = 0; i < image.rows(); ++i)
        for (size_t j = 0; j < image.cols(); ++j)
            images_file_.read(reinterpret_cast<char *>(&image(i, j)), 1);
}

const LabelledImage &const_iterator::operator*() const { return labelled_image_; }

const_iterator &const_iterator::operator++() {
    if (index_ < owner_.size() - 1)
        read_image();

    ++index_;

    return *this;
}

bool const_iterator::operator!=(const const_iterator &other) const {
    return &owner_ != &other.owner_ || index_ != other.index_;
}

ostream &operator<<(ostream &os, const LabelledImage &labelled_image) {
    os << "Label: " << static_cast<int>(labelled_image.label) << "\n";

    const Matrix<uint8_t> &image = labelled_image.image;

    for (size_t i = 0; i < image.rows(); ++i) {
        for (size_t j = 0; j < image.cols(); ++j) {
            int value = static_cast<int>(image(i, j));
            string s = to_string(value);

            os << s << string(4 - s.length(), ' ');
        }
        os << '\n';
    }
    os << '\n';

    return os;
}

ostream &operator<<(ostream &os, const Images &images) {
    os << "Images\n";
    os << "Size: " << images.size() << "\n";
    os << "Rows: " << images.rows() << "\n";
    os << "Cols: " << images.cols() << "\n\n";

    size_t i = 1;

    for (const auto &labelled_image : images) {
        os << "Index: " << i << "\n";
        os << labelled_image;
        ++i;
    }

    return os;
}
