#ifndef HISTOGRAM_EQUALIZATION_UTILS_CUH
#define HISTOGRAM_EQUALIZATION_UTILS_CUH

#include <string>

#define GRAYSCALE_RANGE 256
#define HISTOGRAM_WIDTH GRAYSCALE_RANGE
#define HISTOGRAM_HEIGHT 150

template<class T>
void computeCumulativeSum(T *output, T *input, int size) {
    output[0] = input[0];

    for (int i = 1; i < size; ++i) {
        output[i] = output[i - 1] + input[i];
    }
}

cudaError_t runKernelWithCuda(void(*kernel)(void *, const void *, const void *),
                              void *out,
                              const void *in1,
                              const void *in2,
                              int outSizeInBytes,
                              int inSize1InBytes,
                              int inSize2InBytes,
                              int rows,
                              int cols,
                              int sharedMemorySizeInBytes);

#endif //HISTOGRAM_EQUALIZATION_UTILS_CUH