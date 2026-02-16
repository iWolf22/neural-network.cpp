FROM ubuntu:noble

RUN apt-get update \
    && apt-get install -y \
       build-essential \
       cmake \
       ninja-build \
       git \
       curl \
       zip \
       unzip \
       tar \
       pkg-config \
       plantuml \
       software-properties-common \
    && add-apt-repository -y ppa:bkryza/clang-uml \
    && apt-get update \
    && apt-get install -y clang-uml \
    && rm -rf /var/lib/apt/lists/*

# Install vcpkg
RUN git clone https://github.com/Microsoft/vcpkg.git /opt/vcpkg && \
    /opt/vcpkg/bootstrap-vcpkg.sh

# Set vcpkg environment variables
ENV VCPKG_ROOT=/opt/vcpkg
ENV PATH="${VCPKG_ROOT}:${PATH}"

# Set working directory
WORKDIR /workspace

# Let the container idle
CMD ["/bin/bash"]
