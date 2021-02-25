// vectorizing addition
// indexing threads
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <vector>

using std::begin;
using std::copy;
using std::cout;
using std::end;
using std::endl;
using std::generate;
using std::vector;

// CUDA kernel for vector addition
__global__ void vectorAdd(int* a, int* b, int* c, int N) {
    // calculate global ID thread
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    // bounds check
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    // array size of 2^16
    constexpr int N = 1 << 16;
    size_t bytes = sizeof(int) * N;

    // vectors for holding host-side data
    vector<int> a(N);
    vector<int> b(N);
    vector<int> c(N);

    // initialize numbers in each array
    generate(begin(a), end(a), []() {return rand() % 100; });
    generate(begin(b), end(b), []() {return rand() % 100; });

    // allocate memory on device
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    //copy data from the host to the device (CPU -> GPU)
    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);
    // dont need c vector, will be created on GPU

    // threads per CTA (1024 threads per CTA)
    int NUM_THREADS = 1 << 10;

    // CTAs per grid
    // we need to launch at least as many threads as we have elements
    // This equation pads and extra CTA to the grid if N cannot evenly be divided
    // by NUM_THREADS (e.g. N = 1025, NUM_THREADS = 1024)
    int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

    // Launch the kerned on the GPU
    // kernel calls are asynchronous (The CPU Program cintinues execution after call
    // but not necessarily before the kernel finishes)
    vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N);

    // Copy Sum vector from device to host
    // cudaMemcpy is a synchronous operation, and waits for the prior kernel
    // launch to complete (both go to the default stream in this case).
    // therefore, this cudaMemcpy acts as both a memcpy and synchronization barrier
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    // Check result for errors
//    verify)results(a,b,c)

}