#include <iostream>
#include <cstdlib>
#include <cassert>

// Add a scalar to the vector
__global__ void vadd(int *const v, int const a, size_t const len) {
    const unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int gsize = gridDim.x * blockDim.x;

    for (size_t i = gid; i < len; i += gsize) {
        v[i] += a;
    }
}

int main() {
    // Vector length
    constexpr size_t LEN = 100'000;

    // GPU kernel parameters
    constexpr unsigned int grid = 160;
    constexpr unsigned int block = 1024;

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
    vadd<<<grid, block>>>(data, 1, LEN);

    // Wait for the GPU kernel to finish execution
    cudaDeviceSynchronize();

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
