#ifndef HISTOGRAM_EQUALIZATION_UTILS_CUH
#define HISTOGRAMEQUALIZATION_UTILS_H

#include <string>

#define GRAYSCALE_RANGE 256
#define HISTOGRAM_WIDTH GRAYSCALE_RANGE
#define HISTOGRAM_HEIGHT 150

template<class T>
T findMax(T *input, int size) {
    T max = input[0];

    for (int i = 1; i < size; ++i) {
        if (max < input[i]) {
            max = input[i];
        }
    }

    return max;
}

template<class T>
void computeCumulativeSum(T *output, T *input, int size) {
    output[0] = input[0];

    for (int i = 1; i < size; ++i) {
        output[i] = output[i - 1] + input[i];
    }
}

void plotHistogram(const std::string &name, int *histogram);

#endif //HISTOGRAM_EQUALIZATION_UTILS_CUH