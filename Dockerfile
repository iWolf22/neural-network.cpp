FROM ubuntu:resolute

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
