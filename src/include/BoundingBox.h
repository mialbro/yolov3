#pragma once

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include <torch/script.h>
#include <torch/torch.h>

class BoundingBox {
public:
    BoundingBox(torch::Tensor, const cv::Mat&);
private:
    cv::Mat image;
    double prob;
    int x, y, w, h;
    std::vector<double> class_dist;
};