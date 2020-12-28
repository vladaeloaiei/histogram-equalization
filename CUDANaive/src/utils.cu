#include "utils.cuh"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

cudaError_t runKernelWithCuda(void(*kernel)(void *, const void *, const void *),
                              void *out,
                              const void *in1,
                              const void *in2,
                              int outSizeInBytes,
                              int inSize1InBytes,
                              int inSize2InBytes,
                              int rows,
                              int cols) {
    void *dev_in1 = nullptr;
    void *dev_in2 = nullptr;
    void *dev_out = nullptr;
    cudaError_t cudaStatus;

    auto start = std::chrono::high_resolution_clock::now();

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaCalloc((void **) &dev_in1, inSize1InBytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaCalloc((void **) &dev_in2, inSize2InBytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaCalloc((void **) &dev_out, outSizeInBytes);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_in1, in1, inSize1InBytes, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_in2, in2, inSize2InBytes, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

//    auto finish = std::chrono::high_resolution_clock::now();
//    std::cout << "Transfer Host->Device GPU : "
//              << std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() << "microsec\n";

    // Launch a kernel on the GPU with one thread for each element.
    kernel <<<rows, cols >>>(dev_out, dev_in1, dev_in2);

    start = std::chrono::high_resolution_clock::now();

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(out, dev_out, outSizeInBytes, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

//    finish = std::chrono::high_resolution_clock::now();
//    std::cout << "Transfer Device->Host GPU : "
//              << std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() << "microsec\n";

    Error:
    cudaFree(dev_in1);
    cudaFree(dev_in2);
    cudaFree(dev_out);

    return cudaStatus;
}