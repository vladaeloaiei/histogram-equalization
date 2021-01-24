#include "contrast.cuh"
#include "utils.cuh"
#include <cmath>

#define BATCH_DIM 32
#define STREAM_COUNT BATCH_DIM

__global__ static void computeHistogram_kernel(int *histogram, const uchar *inputPixels, int size) {
    int start = (blockDim.x * blockIdx.x + threadIdx.x) * BATCH_DIM;
    int stop = start + BATCH_DIM;

    for (int i = start; (i < stop) && (i < size); ++i) {
        atomicAdd(&histogram[inputPixels[i]], 1);
    }
}

static cudaError_t computeHistogram(int **dev_outputHistogram, int histogramRange, uchar **dev_inputPixels, uchar *inputPixels, int rows, int cols) {
    cudaError_t cudaStatus;
    cudaStream_t streams[STREAM_COUNT];
    int blocks = ceil((double) rows / STREAM_COUNT);
    int threads = ceil((double) cols / BATCH_DIM);
    int size = 0;

    cudaStatus = cudaCalloc(dev_outputHistogram, histogramRange * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "outputHistogram cudaCalloc failed!\n");
        return cudaStatus;
    }

    cudaStatus = cudaMalloc(dev_inputPixels, rows * cols * sizeof(uchar));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "dev_inputPixels cudaMalloc failed!\n");
        return cudaStatus;
    }

    for (int i = 0; i < STREAM_COUNT; ++i) {
        cudaStreamCreate(&streams[i]);

        if (i * blocks < rows) {
            size = (((i + 1) * blocks * cols) < (rows * cols) ? (blocks * cols) : ((rows - i * blocks) * cols));

            cudaStatus = cudaMemcpyAsync(&dev_inputPixels[0][i * blocks * cols],
                                         &inputPixels[i * blocks * cols],
                                         size * sizeof(uchar),
                                         cudaMemcpyHostToDevice,
                                         streams[i]);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "dev_inputPixels[%d] cudaMemcpyAsync failed!\n", i);
                return cudaStatus;
            }

            computeHistogram_kernel <<<blocks, threads, 0, streams[i]>>>(dev_outputHistogram[0], &dev_inputPixels[0][i * blocks * cols], size);
        }
    }

    // cudaDeviceSynchronize waits for the kernel to finish
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching computeHistogram_kernel!\n", cudaStatus);
    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to launch computeHistogram_kernel: %s\n", cudaGetErrorString(cudaStatus));
    }

    for (int i = 0; i < STREAM_COUNT; ++i) {
        cudaStreamDestroy(streams[i]);
    }

    return cudaStatus;
}

__global__ static void computeCumulativeSum_kernel(int *cumulativeHistogram, const int *histogram) {
    extern __shared__ int sharedHistogram[];
    unsigned int id = threadIdx.x;

    sharedHistogram[id] = histogram[id];

    __syncthreads();

    for (int i = 0; i <= id; ++i) {
        cumulativeHistogram[id] += sharedHistogram[i];
    }
}

static cudaError_t computeCumulativeSum(int **dev_outputCumulativeHistogram, int *dev_histogram, int histogramRange) {
    cudaError_t cudaStatus;

    cudaStatus = cudaCalloc(dev_outputCumulativeHistogram, histogramRange * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "dev_outputCumulativeHistogram cudaCalloc failed!\n");
        return cudaStatus;
    }

    computeCumulativeSum_kernel<<<1, histogramRange, histogramRange * sizeof(int)>>>(dev_outputCumulativeHistogram[0], dev_histogram);

    // cudaDeviceSynchronize waits for the kernel to finish
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching computeCumulativeSum_kernel!\n", cudaStatus);
    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to launch computeHistogram_kernel: %s\n", cudaGetErrorString(cudaStatus));
    }

    return cudaStatus;
}

__global__ static void equalizeHistogram_kernel(int *pixelMapping, const int *cumulativeHistogram, int histogramRange) {
    extern __shared__ int sharedCumulativeHistogram[];
    unsigned int cumulativeHistogramSize = histogramRange;
    unsigned int id = threadIdx.x;

    sharedCumulativeHistogram[id] = cumulativeHistogram[id];

    __syncthreads();

    pixelMapping[id] = (int) (sharedCumulativeHistogram[id] * (cumulativeHistogramSize - 1) /
                              sharedCumulativeHistogram[cumulativeHistogramSize - 1]);
}


static cudaError_t equalizeHistogram(int **dev_outputPixelMapping, const int *dev_cumulativeHistogram, int histogramRange) {
    cudaError_t cudaStatus;

    cudaStatus = cudaCalloc(dev_outputPixelMapping, histogramRange * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "dev_outputPixelMapping cudaCalloc failed!\n");
        return cudaStatus;
    }

    equalizeHistogram_kernel<<<1, histogramRange, histogramRange * sizeof(int)>>>(dev_outputPixelMapping[0], dev_cumulativeHistogram, histogramRange);

    // cudaDeviceSynchronize waits for the kernel to finish
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching equalizeHistogram_kernel!\n", cudaStatus);
    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to launch equalizeHistogram_kernel: %s\n", cudaGetErrorString(cudaStatus));
    }

    return cudaStatus;
}

__global__ static void applyHistogramEqualizationOnImage_kernel(uchar *pixels, const int *pixelMapping, int size) {
    int start = (blockDim.x * blockIdx.x + threadIdx.x) * BATCH_DIM;
    int stop = start + BATCH_DIM;

    for (int i = start; (i < stop) && (i < size); ++i) {
        pixels[i] = pixelMapping[pixels[i]];
    }
}

static cudaError_t applyHistogramEqualizationOnImage(uchar *outputPixels, uchar *dev_inputPixels, int *dev_pixelMapping, int rows, int cols) {
    cudaError_t cudaStatus;
    cudaStream_t streams[STREAM_COUNT];
    int blocks = ceil((double) rows / STREAM_COUNT);
    int threads = ceil((double) cols / BATCH_DIM);
    int size = 0;

    for (int i = 0; (i < STREAM_COUNT) && ((i * blocks) < rows); ++i) {
        cudaStreamCreate(&streams[i]);

        if (i * blocks < rows) {
            size = (((i + 1) * blocks * cols) < (rows * cols) ? (blocks * cols) : ((rows - i * blocks) * cols));

            applyHistogramEqualizationOnImage_kernel<<<blocks, threads, 0, streams[i]>>>(&dev_inputPixels[i * blocks * cols], dev_pixelMapping, size);
        }
    }

    // cudaDeviceSynchronize waits for the kernel to finish
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching applyHistogramEqualizationOnImage_kernel!\n", cudaStatus);
    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to launch applyHistogramEqualizationOnImage_kernel: %s\n", cudaGetErrorString(cudaStatus));
    }

    cudaStatus = cudaMemcpy(outputPixels,
                            dev_inputPixels,
                            rows * cols * sizeof(uchar),
                            cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "outputPixels cudaMemcpyAsync failed!\n");
        return cudaStatus;
    }

    for (int i = 0; (i < STREAM_COUNT) && ((i * blocks) < rows); ++i) {
        cudaStreamDestroy(streams[i]);
    }

    return cudaStatus;
}

cudaError_t cudaEnhanceContrast(uchar *outputPixels,
                                uchar **dev_inputPixels,
                                uchar *inputPixels,
                                int rows,
                                int cols,
                                int **dev_outputHistogram,
                                int **dev_outputCumulativeHistogram,
                                int **dev_outputPixelMapping,
                                int histogramRange) {
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaSuccess != cudaStatus) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?\n");
        return cudaStatus;
    }

    //Compute histogram
    cudaStatus = computeHistogram(dev_outputHistogram, histogramRange, dev_inputPixels, inputPixels, rows, cols);
    if (cudaSuccess != cudaStatus) {
        fprintf(stderr, "Failed to computeHistogram\n");
        return cudaStatus;
    }

    //Compute cumulativeHistogram
    cudaStatus = computeCumulativeSum(dev_outputCumulativeHistogram, dev_outputHistogram[0], histogramRange);
    if (cudaSuccess != cudaStatus) {
        fprintf(stderr, "Failed to computeCumulativeSum\n");
        return cudaStatus;
    }

    //Equalize histogram
    cudaStatus = equalizeHistogram(dev_outputPixelMapping, dev_outputCumulativeHistogram[0], histogramRange);
    if (cudaSuccess != cudaStatus) {
        fprintf(stderr, "Failed to equalizeHistogram\n");
        return cudaStatus;
    }

    //Enhance contrast
    cudaStatus = applyHistogramEqualizationOnImage(outputPixels, dev_inputPixels[0], dev_outputPixelMapping[0], rows, cols);
    if (cudaSuccess != cudaStatus) {
        fprintf(stderr, "Failed to applyHistogramEqualizationOnImage\n");
        return cudaStatus;
    }

    return cudaStatus;
}


cudaError_t enhanceContrast(cv::Mat &outputImage, const cv::Mat &inputImage, int histogramRange) {
    uchar *dev_inputPixels;
    int *dev_histogram = nullptr;
    int *dev_cumulativeHistogram = nullptr;
    int *dev_pixelMapping = nullptr;
    cudaError_t cudaStatus;

    //Enhance contrast
    cudaStatus = cudaEnhanceContrast(outputImage.data,
                                     &dev_inputPixels,
                                     inputImage.data,
                                     inputImage.rows,
                                     inputImage.cols,
                                     &dev_histogram,
                                     &dev_cumulativeHistogram,
                                     &dev_pixelMapping,
                                     histogramRange);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to enhance contrast. Status = %s\n", cudaGetErrorString(cudaStatus));
    }

    //free resources
    cudaFree(dev_inputPixels);
    cudaFree(dev_histogram);
    cudaFree(dev_cumulativeHistogram);
    cudaFree(dev_pixelMapping);

    return cudaStatus;
}
