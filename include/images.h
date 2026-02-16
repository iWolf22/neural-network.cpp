#pragma once

#include "matrix.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <ostream>
#include <utility>

enum class ImageType {
    TRAINING,
    TESTING,
};

enum class FileType { IMAGES, LABELS };

struct ImagesEndTag;

class Images {
    size_t size_{};
    uint32_t rows_{};
    uint32_t cols_{};

    std::filesystem::path images_filename_;
    std::filesystem::path labels_filename_;

    void get_mnist_filepaths(ImageType image_type,
                             const std::filesystem::path &data_dir);

  public:
    struct LabelledImage {
        Matrix<uint8_t> image;
        uint8_t label;
    };

    class const_iterator {
        const Images &owner_;

        std::ifstream images_file_;
        std::ifstream labels_file_;

        LabelledImage labelled_image_;

        size_t index_;

        void skip_headers();
        void read_image();

      public:
        const_iterator(const Images &owner);
        const_iterator(const Images &owner, ImagesEndTag);

        const LabelledImage &operator*() const;

        const_iterator &operator++();

        bool operator!=(const const_iterator &other) const;
    };

    Images(ImageType image_type,
           std::filesystem::path data_dir = "/workspace/data");

    std::ifstream open_file(FileType file_type) const;

    size_t size() const;
    uint32_t rows() const;
    uint32_t cols() const;

    const_iterator begin() const;
    const_iterator end() const;
};

std::ostream &operator<<(std::ostream &os,
                         const Images::LabelledImage &labelled_image);
std::ostream &operator<<(std::ostream &os, const Images &images);
