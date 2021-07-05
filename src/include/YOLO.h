#pragma once

#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include <torch/script.h>
#include <torch/torch.h>


//#include "BoundingBox.h"
class BoundingBox;

class YOLO {
public:
    YOLO(const std::string&);
    std::vector<BoundingBox> predict(const cv::Mat&);
private:
    torch::jit::script::Module model;
    torch::Device device = (torch::kCPU);
};