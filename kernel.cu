#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <vector>
#include <iostream>
#include <chrono>
__global__ void countKernel(const int* inputVector, int* countArray, int NumberOfElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NumberOfElements) {
        atomicAdd(&countArray[inputVector[idx]], 1);  // Atomic operation to avoid race condition
    }
}

__global__ void placeKernel(const int* inputVector, const int* countArray, int* outputArray, int NumberOfElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NumberOfElements) {
        int value = inputVector[idx];
        int pos = atomicSub((int*)&countArray[value], 1) - 1;  // Get position and decrement count
        outputArray[pos] = value;
    }
}


#define threadsPerBlock 1024
void countingSortCUDA(int upperBound, int NumberOfElements, std::vector<int>& inputVector) {
    // Step 1: Define GPU memory pointers
    int* d_inputVector, * d_countArray, * d_outputArray;
    const int range = upperBound + 1;

    // Step 2: Allocate device memory
    cudaMalloc(&d_inputVector, NumberOfElements * sizeof(int));
    cudaMalloc(&d_outputArray, NumberOfElements * sizeof(int));
    cudaMalloc(&d_countArray, range * sizeof(int));

    // Step 3: Initialize countArray to zero
    cudaMemset(d_countArray, 0, range * sizeof(int));

    // Step 4: Copy input data to device
    cudaMemcpy(d_inputVector, inputVector.data(), NumberOfElements * sizeof(int), cudaMemcpyHostToDevice);

    // Step 5: Launch kernel to count occurrences
    int blocksPerGrid = (NumberOfElements + threadsPerBlock - 1) / threadsPerBlock;
    countKernel << <blocksPerGrid, threadsPerBlock >> > (d_inputVector, d_countArray, NumberOfElements);
    cudaDeviceSynchronize();

    // Step 6: Perform cumulative sum (prefix-sum) on the countArray using thrust
    thrust::device_ptr<int> thrust_countArray(d_countArray);
    thrust::inclusive_scan(thrust_countArray, thrust_countArray + range, thrust_countArray);

    // Step 7: Launch kernel to place elements in the correct output positions
    placeKernel << <blocksPerGrid, threadsPerBlock >> > (d_inputVector, d_countArray, d_outputArray, NumberOfElements);
    cudaDeviceSynchronize();

    // Step 8: Copy sorted data back to host
    cudaMemcpy(inputVector.data(), d_outputArray, NumberOfElements * sizeof(int), cudaMemcpyDeviceToHost);

    // Step 9: Free GPU memory
    cudaFree(d_inputVector);
    cudaFree(d_countArray);
    cudaFree(d_outputArray);
}

int main() {
    int upperBound = 9;  // Example: max value in array
    std::vector<int> inputVector = { 4, 2, 2, 8, 3, 3, 1, 9, 0, 7, 5, 4, 6 };

    std::cout << "Input Array: ";
    for (int v : inputVector) std::cout << v << " ";
    std::cout << std::endl;

    // Call CUDA version of counting sort
    countingSortCUDA(upperBound, inputVector.size(), inputVector);

    std::cout << "Sorted Array: ";
    for (int v : inputVector) std::cout << v << " ";
    std::cout << std::endl;

    return 0;
}
