#include "utils.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

void plotHistogram(const std::string &name, int *histogram, int size) {
    cv::Mat histogramImage(HISTOGRAM_HEIGHT, HISTOGRAM_WIDTH, CV_8U);
    int max = findMax(histogram, size);

    for (int i = 0; i < HISTOGRAM_WIDTH; ++i) {
        line(histogramImage,
             cv::Point(i, HISTOGRAM_HEIGHT),
             cv::Point(i, (int) (HISTOGRAM_HEIGHT * (max - histogram[i]) / max)),
             cv::Scalar(0, 0, 0),
             2,
             8,
             0);
    }

    cv::imshow(name, histogramImage);
}