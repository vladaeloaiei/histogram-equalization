#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "contrast.h"
#include "utils.h"

#define LOW_CONTRAST_LENA "../input/lac.png"

int main() {
    cv::Mat image = imread(LOW_CONTRAST_LENA, cv::IMREAD_GRAYSCALE);
    cv::Mat enhancedImage(image.size(), CV_8U);

    auto start = std::chrono::high_resolution_clock::now();

    enhanceContrast(enhancedImage, image, GRAYSCALE_RANGE);

    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << "Histogram equalization run time CPU : "
              << std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() << "microsec\n";


    //PLOTTING
    imshow("Lena", image);
    imshow("Lena enhanced", enhancedImage);

//    plotHistogram("Lena hist", hist);
//    plotHistogram("Lena cumulativeHist", cumulativeHist);
//    plotHistogram("Lena hist enhanced", histEnhanced);
//    plotHistogram("Lena cumulativeHist enhanced", cumulativeHistEnhanced);

    cv::waitKey(0);
    return 0;
}
