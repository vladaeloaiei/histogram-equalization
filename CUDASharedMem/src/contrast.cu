#include "contrast.cuh"
#include "utils.cuh"

__global__ static void computeHistogram_kernel(void *out,
                                               const void *in1,
                                               const void *in2) {
    int *histogram = (int *) out;
    const unsigned char *image = (unsigned char *) in1;
    unsigned int cols = blockDim.x;
    unsigned int i = blockIdx.x;
    unsigned int j = threadIdx.x;

    atomicAdd(histogram + image[i * cols + j], 1);
}

static void computeHistogram(int *output, const cv::Mat &input, int histogramRange) {
    cudaError_t cudaStatus = runKernelWithCuda(computeHistogram_kernel,
                                               (void *) output,
                                               (const void *) input.data,
                                               nullptr,
                                               (int) (histogramRange * sizeof(int)),
                                               (int) (input.cols * input.rows * sizeof(unsigned char)),
                                               0,
                                               input.rows,
                                               input.cols,
                                               0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
    }
}

__global__ static void equalizeHistogram_kernel(void *out,
                                                const void *in1,
                                                const void *in2) {
    extern __shared__ int sharedCumulativeHistogram[];
    int *pixelMapping = (int *) out;
    const int *cumulativeHistogram = (int *) in1;
    unsigned int cumulativeHistogramSize = blockDim.x;
    unsigned int i = threadIdx.x;

    sharedCumulativeHistogram[i] = cumulativeHistogram[i];

    __syncthreads();

    //pixelMappingSize == cumulativeHistogramSize
    pixelMapping[i] = (int) (sharedCumulativeHistogram[i] * (cumulativeHistogramSize - 1) /
                             sharedCumulativeHistogram[cumulativeHistogramSize - 1]);
}


static void equalizeHistogram(int *pixelMapping, const int *cumulativeHistogram, int size) {
    cudaError_t cudaStatus = runKernelWithCuda(equalizeHistogram_kernel,
                                               (void *) pixelMapping,
                                               (const void *) cumulativeHistogram,
                                               nullptr,
                                               (int) (size * sizeof(int)),
                                               (int) (size * sizeof(int)),
                                               0,
                                               1,
                                               size,
                                               (int) (size * sizeof(int)));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
    }
}

__global__ static void applyHistogramEqualizationOnImage_kernel(void *out,
                                                                const void *in1,
                                                                const void *in2) {
    unsigned char *outputImage = (unsigned char *) out;
    const unsigned char *inputImage = (unsigned char *) in1;
    const int *mapping = (int *) in2;
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    outputImage[index] = mapping[inputImage[index]];
}

static void applyHistogramEqualizationOnImage(cv::Mat &output,
                                              const cv::Mat &input,
                                              const int *mapping,
                                              int histogramRange) {
    cudaError_t cudaStatus = runKernelWithCuda(applyHistogramEqualizationOnImage_kernel,
                                               (void *) output.data,
                                               (const void *) input.data,
                                               (const void *) mapping,
                                               (int) (output.rows * output.cols * sizeof(unsigned char)),
                                               (int) (output.rows * output.cols * sizeof(unsigned char)),
                                               (int) (histogramRange * sizeof(int)),
                                               output.rows,
                                               output.cols,
                                               0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
    }
}

void enhanceContrast(cv::Mat &outputImage, const cv::Mat &inputImage, int histogramRange) {
    int *histogram = new int[histogramRange]();
    int *cumulativeHistogram = new int[histogramRange]();
    int *pixelMapping = new int[histogramRange]();

    //Compute histogram
    computeHistogram(histogram, inputImage, histogramRange);
    //Compute cumulativeHistogram
    computeCumulativeSum(cumulativeHistogram, histogram, histogramRange);
    //Equalize histogram
    equalizeHistogram(pixelMapping, cumulativeHistogram, histogramRange);
    //Enhance contrast
    applyHistogramEqualizationOnImage(outputImage, inputImage, pixelMapping, histogramRange);

    //free resources
    delete[] histogram;
    delete[] cumulativeHistogram;
    delete[] pixelMapping;
}

