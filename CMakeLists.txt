cmake_minimum_required(VERSION 3.20)

project(neural-network
    VERSION 1.0.0
    LANGUAGES CXX
)

# Set C++ standard to C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags: enable all warnings and debug symbols
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g")

# Source files
set(SOURCES
    src/main.cpp
    src/utils/images.cpp
)

# Neural network executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Test executable
find_package(GTest CONFIG REQUIRED)
add_executable(test_main tests/test_main.cpp)
target_link_libraries(test_main PRIVATE GTest::gtest GTest::gtest_main)
enable_testing()
add_test(NAME MainTest COMMAND test_main)
