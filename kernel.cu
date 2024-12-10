#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <chrono>
#include <curand_kernel.h>
#include <cub/cub.cuh> // Include the CUB library header
#include <cassert> // For assert

#define threadsperblock 1024

// Kernel to initialize CURAND states
__global__ void InitCurandStates(curandState* states, unsigned long seed, int NumberOfElements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NumberOfElements) {
        curand_init(seed, tid, 0, &states[tid]);
    }
}

// Kernel to generate random numbers
__global__ void GenerateRandomArrayKernel(int* d_array, curandState* states, int lowerBound, int upperBound, int NumberOfElements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NumberOfElements) {
        curandState localState = states[tid]; // Use pre-initialized state
        float randomValue = curand_uniform(&localState); // Generate random float in range (0, 1]
        d_array[tid] = lowerBound + (int)((upperBound - lowerBound + 1) * randomValue);
        states[tid] = localState; // Save updated state
    }
}

// Function to generate a random array using CUDA
int* CreateRandomArray(int NumberOfElements, int lowerBound, int upperBound) {
    int* d_array;
    int* h_array = new int[NumberOfElements];
    curandState* d_states;

    // Allocate device memory
    cudaMalloc(&d_array, sizeof(int) * NumberOfElements);
    cudaMalloc(&d_states, sizeof(curandState) * NumberOfElements);

    // Configure kernel
    int blocksPerGrid = (NumberOfElements + threadsperblock - 1) / threadsperblock;
    unsigned long seed = time(0);

    // Initialize CURAND states
    InitCurandStates << <blocksPerGrid, threadsperblock >> > (d_states, seed, NumberOfElements);

    // Generate random numbers using pre-initialized states
    GenerateRandomArrayKernel << <blocksPerGrid, threadsperblock >> > (d_array, d_states, lowerBound, upperBound, NumberOfElements);

    // Copy the generated random numbers back to host memory
    cudaMemcpy(h_array, d_array, sizeof(int) * NumberOfElements, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_array);
    cudaFree(d_states);

    return h_array;
}

// Kernel to initialize an array
__global__ void InitializeArray(int* array, int size, int value) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        array[tid] = value;
    }
}

// Kernel to count occurrences of each element
__global__ void CountOccurrences(int* inputArray, int* countArray, int NumberOfElements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NumberOfElements) {
        atomicAdd(&countArray[inputArray[tid]], 1);
    }
}

// Kernel to place elements in the correct position
__global__ void PlaceElements(int* inputArray, int* outputArray, int* countArray, int NumberOfElements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NumberOfElements) {
        int value = inputArray[tid];
        int position = atomicSub(&countArray[value], 1) - 1;
        outputArray[position] = value;
    }
}

// Counting Sort function using CUDA
void CountingSortGPU(int upperBound, int NumberOfElements, int* inputArray) {
    int* d_inputArray;
    int* d_outputArray;
    int* d_countArray;

    int range = upperBound + 1;

    // Allocate device memory
    cudaMalloc(&d_inputArray, sizeof(int) * NumberOfElements);
    cudaMalloc(&d_outputArray, sizeof(int) * NumberOfElements);
    cudaMalloc(&d_countArray, sizeof(int) * range);

    auto start = std::chrono::high_resolution_clock::now();


    // Copy input data to device
    cudaMemcpy(d_inputArray, inputArray, sizeof(int) * NumberOfElements, cudaMemcpyHostToDevice);


    // Initialize the count array on the device
    int blocksPerGridCount = (range + threadsperblock - 1) / threadsperblock;
    InitializeArray << <blocksPerGridCount, threadsperblock >> > (d_countArray, range, 0);

    // Count occurrences of each element
    int blocksPerGridInput = (NumberOfElements + threadsperblock - 1) / threadsperblock;
    CountOccurrences << <blocksPerGridInput, threadsperblock >> > (d_inputArray, d_countArray, NumberOfElements);

    // Compute cumulative counts using CUB InclusiveSum
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Get temporary storage size for CUB InclusiveSum
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_countArray, d_countArray, range);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Perform cumulative sum
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_countArray, d_countArray, range);

    // Place elements in the correct position
    PlaceElements << <blocksPerGridInput, threadsperblock >> > (d_inputArray, d_outputArray, d_countArray, NumberOfElements);



    // Copy sorted array back to host
    cudaMemcpy(inputArray, d_outputArray, sizeof(int) * NumberOfElements, cudaMemcpyDeviceToHost);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;


    std::cout << "Time taken to sort using Counting Sort on GPU: " << duration.count() << " ms" << std::endl;

    // Free device memory
    cudaFree(d_temp_storage);
    cudaFree(d_inputArray);
    cudaFree(d_outputArray);
    cudaFree(d_countArray);
}

int main() {
    int NumberOfElements = 20e8; // Change this value for variable-sized arrays
    int lowerBound = 0;
    int upperBound = 9;

    // Generate random array on GPU
    int* inputArray = CreateRandomArray(NumberOfElements, lowerBound, upperBound);


    // Perform Counting Sort
    CountingSortGPU(upperBound, NumberOfElements, inputArray);



    // Assertion to verify the array is sorted
    for (int i = 1; i < NumberOfElements; ++i) {
        assert(inputArray[i - 1] <= inputArray[i] && "Array is not sorted!");
    }

    std::cout << "Array is sorted correctly!" << std::endl;

    // Free host memory
    delete[] inputArray;

    return 0;
}