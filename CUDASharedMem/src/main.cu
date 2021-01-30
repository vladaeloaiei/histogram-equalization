#pragma once
//CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//User defined
#include <iostream>
#include "../inc/utils.cuh"
#include "../inc/contrast.cuh"

//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#define LOW_CONTRAST_LENA "../input/1020x674.png"

using namespace cv;

int main() {
    cv::Mat image = imread(LOW_CONTRAST_LENA, cv::IMREAD_GRAYSCALE);
    cv::Mat enhancedImage(image.size(), CV_8U);

    auto start = std::chrono::high_resolution_clock::now();

    //for (int i = 0; i < 100; ++i) {
        enhanceContrast(enhancedImage, image, GRAYSCALE_RANGE);
    //}

    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << "Histogram equalization run time GPU optimized : "
              << std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() << "microsec\n";

    //PLOTTING
    imshow("Lena", image);
    imshow("Lena enhanced", enhancedImage);

    cv::waitKey(0);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

