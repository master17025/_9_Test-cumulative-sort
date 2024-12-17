#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <vector>
#include <iostream>
#include <ctime>
#include <chrono>

#define threadsperblock 1024

// Kernel to initialize CURAND states
__global__ void InitCurandStates(curandState* states, long long seed, int NumberOfElements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NumberOfElements) {
        curand_init(seed, tid, 0, &states[tid]);
    }
}

// Kernel to generate random numbers
__global__ void GenerateRandomArrayKernel(int* d_array, curandState* states, int lowerBound, int upperBound, int NumberOfElements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NumberOfElements) {
        curandState localState = states[tid];
        float randomValue = curand_uniform(&localState); // Generate random float
        d_array[tid] = lowerBound + (int)((upperBound - lowerBound + 1) * randomValue);
        states[tid] = localState;
    }
}

int* CreateRandomArray(int NumberOfElements, int lowerBound, int upperBound) {
    int* d_array;
    int* h_array = new int[NumberOfElements];
    curandState* d_states;

    cudaMalloc(&d_array, sizeof(int) * NumberOfElements);
    cudaMalloc(&d_states, sizeof(curandState) * NumberOfElements);

    int blocksPerGrid = (NumberOfElements + threadsperblock - 1) / threadsperblock;
    long long seed = time(0);

    InitCurandStates << <blocksPerGrid, threadsperblock >> > (d_states, seed, NumberOfElements);
    GenerateRandomArrayKernel << <blocksPerGrid, threadsperblock >> > (d_array, d_states, lowerBound, upperBound, NumberOfElements);

    cudaMemcpy(h_array, d_array, sizeof(int) * NumberOfElements, cudaMemcpyDeviceToHost);

    cudaFree(d_array);
    cudaFree(d_states);

    return h_array;
}

// CUDA Counting Sort Kernels
__global__ void countKernel(const int* inputVector, int* countArray, long int NumberOfElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NumberOfElements) {
        atomicAdd(&countArray[inputVector[idx]], 1);
    }
}

__global__ void placeKernel(const int* inputVector, int* countArray, int* outputArray, long int NumberOfElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NumberOfElements) {
        int value = inputVector[idx];
        int pos = atomicSub(&countArray[value], 1) - 1;
        outputArray[pos] = value;
    }
}

void countingSortCUDA(int upperBound, int NumberOfElements, std::vector<int>& inputVector) {
    int* d_inputVector, * d_countArray, * d_outputArray;
    const int range = upperBound + 1;

    cudaMalloc(&d_inputVector, NumberOfElements * sizeof(int));
    cudaMalloc(&d_outputArray, NumberOfElements * sizeof(int));
    cudaMalloc(&d_countArray, range * sizeof(int));



    cudaMemset(d_countArray, 0, range * sizeof(int));
    cudaMemcpy(d_inputVector, inputVector.data(), NumberOfElements * sizeof(int), cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();
    int blocksPerGrid = (NumberOfElements + threadsperblock - 1) / threadsperblock;

    countKernel << <blocksPerGrid, threadsperblock >> > (d_inputVector, d_countArray, NumberOfElements);

    thrust::device_ptr<int> thrust_countArray(d_countArray);
    thrust::inclusive_scan(thrust_countArray, thrust_countArray + range, thrust_countArray);

    placeKernel << <blocksPerGrid, threadsperblock >> > (d_inputVector, d_countArray, d_outputArray, NumberOfElements);


    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Time taken to sort the list: " << duration.count() << " milliseconds" << std::endl;

    cudaMemcpy(inputVector.data(), d_outputArray, NumberOfElements * sizeof(int), cudaMemcpyDeviceToHost);


    cudaFree(d_inputVector);
    cudaFree(d_countArray);
    cudaFree(d_outputArray);
}

int main() {
    const int NumberOfElements = 1L << 28;  // Maximum 21
    const int lowerBound = 1;
    const int upperBound = 1 << 12;

    int* h_randomList = CreateRandomArray(NumberOfElements, lowerBound, upperBound);
    std::vector<int> inputVector(h_randomList, h_randomList + NumberOfElements);

    std::cout << "Sorting " << NumberOfElements << " elements..." << std::endl;

    countingSortCUDA(upperBound, NumberOfElements, inputVector);

    // Verify the array is sorted
    for (int i = 1; i < NumberOfElements - 1; i++) {
        if (!(inputVector[i - 1] <= inputVector[i])) {
            std::cerr << "Array is not sorted correctly!" << std::endl;
            std::cout << "Error at index " << i << ": " << inputVector[i - 1] << " > " << inputVector[i] << std::endl;
            delete[] h_randomList;
            return -1;
        }
    }
    std::cout << "Array is sorted correctly!" << std::endl;

    delete[] h_randomList;
    return 0;
}
