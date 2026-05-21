cmake_minimum_required(VERSION 3.20)

project(neural-network
    VERSION 1.0.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g") # Compiler flags: enable all warnings and debug symbols
set(SOURCES
    src/utils/images.cpp
    src/utils/utilities.cpp
    src/simple/dense_layer.cpp
    src/simple/neural_network.cpp
    src/batched/dense_layer.cpp
    src/batched/neural_network.cpp
)
set(TEST_SOURCES
    tests/utils/matrix_test.cpp
    tests/utils/images_test.cpp
)

# Neural network executable
add_executable(${PROJECT_NAME} src/main.cpp ${SOURCES})
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

find_package(GTest CONFIG REQUIRED)

# GTest executable
add_executable(testing-suite ${SOURCES} ${TEST_SOURCES})
target_include_directories(testing-suite PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(testing-suite PRIVATE GTest::gtest GTest::gtest_main)
enable_testing()
add_test(NAME MainTest COMMAND testing-suite)
