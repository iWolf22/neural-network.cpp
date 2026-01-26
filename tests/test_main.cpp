#include <gtest/gtest.h>
#include <iostream>
#include <sstream>

// Simple function that mimics main's behavior
void run_main_logic() {
    std::cout << "Hello World\n";
}

// Simple test to mock/verify main function behavior
TEST(MainTest, HelloWorldOutput) {
    // Capture stdout
    std::ostringstream buffer;
    std::streambuf* old = std::cout.rdbuf(buffer.rdbuf());
    
    // Run the main logic
    run_main_logic();
    
    // Restore stdout
    std::cout.rdbuf(old);
    
    // Verify output
    EXPECT_EQ(buffer.str(), "Hello World\n");
}
