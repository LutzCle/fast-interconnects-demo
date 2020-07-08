#include <iostream>
#include <cstdlib>
#include <cassert>

// Add a scalar to the vector
void vadd(int *const v, int const a, size_t const len) {

    for (size_t i = 0; i < len; ++i) {
        v[i] += a;
    }
}

int main() {
    // Vector length
    constexpr size_t LEN = 100'000;

    // Allocate vector
    int *data = nullptr;
    data = reinterpret_cast<int *>(malloc(LEN * sizeof(int)));
    if (data == nullptr) {
        std::cerr << "Failed to allocate memory" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Initialize vector with some data
    for (size_t i = 0; i < LEN; ++i) {
        data[i] = i;
    }

    // Call a function to do some work
    vadd(data, 1, LEN);

    // Verify that result is correct
    unsigned long long sum = 0;
    for (size_t i = 0; i < LEN; ++i) {
        sum += data[i];
    }
    assert(sum == (LEN * (LEN + 1)) / 2);

    // Free vector
    free(data);

    std::exit(EXIT_SUCCESS);
}
