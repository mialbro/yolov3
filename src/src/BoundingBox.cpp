#include "BoundingBox.h"

BoundingBox(torch::Tensor result, cont cv::Mat& image): image(image) {
    class_dist.reserve(20);

    int x = result[0].item<int>();
    int y = result[0].item<int>();
    int w = result[0].item<int>();
    int h = result[0].item<int>();

    double prob = result[4].item<double>();

    for (int i = 5; i < 25; i++) {
        class_dist[i] = result[i].item<int>();
    }
}