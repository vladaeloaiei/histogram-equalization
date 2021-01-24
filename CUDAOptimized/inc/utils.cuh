#ifndef HISTOGRAM_EQUALIZATION_UTILS_CUH
#define HISTOGRAM_EQUALIZATION_UTILS_CUH

#include <string>

#define GRAYSCALE_RANGE 256
#define HISTOGRAM_WIDTH GRAYSCALE_RANGE
#define HISTOGRAM_HEIGHT 150

template<class T>
static __inline__ __host__ cudaError_t cudaCalloc(T **devPtr, size_t size) {
    cudaError_t cudaStatus = cudaMalloc((void **) devPtr, size);

    if (cudaStatus == cudaSuccess) {
        cudaStatus = cudaMemset(*devPtr, 0, size);
    }

    return cudaStatus;
}

#endif //HISTOGRAM_EQUALIZATION_UTILS_CUH