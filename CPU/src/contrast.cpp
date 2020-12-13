#include "contrast.h"
#include "utils.h"

static void computeHistogram(int *output, const cv::Mat &input) {
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            output[input.at<uchar>(i, j)]++;
        }
    }
}

static void equalizeHistogram(int *pixelMapping, const int *cumulativeHistogram, int size) {
    for (int i = 0; i < size; ++i) {
        pixelMapping[i] = (int) (cumulativeHistogram[i] * (size - 1) / cumulativeHistogram[size - 1]);
    }
}

static void applyHistogramEqualizationOnImage(cv::Mat &output, const cv::Mat &input, const int *mapping) {
    for (int i = 0; i < output.rows; ++i) {
        for (int j = 0; j < output.cols; ++j) {
            output.at<uchar>(i, j) = mapping[input.at<uchar>(i, j)];
        }
    }
}

void enhanceContrast(cv::Mat &outputImage, const cv::Mat &inputImage, int histogramRange) {
    int* histogram = new int[histogramRange]();
    int* cumulativeHistogram = new int[histogramRange]();
    int* pixelMapping = new int[histogramRange]();

    //Compute histogram
    computeHistogram(histogram, inputImage);
    //Compute cumulativeHistogram
    computeCumulativeSum(cumulativeHistogram, histogram, histogramRange);
    //Equalize histogram
    equalizeHistogram(pixelMapping, cumulativeHistogram, histogramRange);
    //Enhance contrast
    applyHistogramEqualizationOnImage(outputImage, inputImage, pixelMapping);

    //free resources
    delete[] histogram;
    delete[] cumulativeHistogram;
    delete[] pixelMapping;
}

