#pragma once

#ifndef HISTOGRAMEQUALIZATION_CONTRAST_CUH
#define HISTOGRAMEQUALIZATION_CONTRAST_CUH

#include <opencv2/core/core.hpp>

void enhanceContrast(cv::Mat &outputImage, const cv::Mat &inputImage, int histogramRange);

#endif //HISTOGRAMEQUALIZATION_CONTRAST_CUH