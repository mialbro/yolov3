#include "YOLO.h"

void YOLO::YOLO(const std::string& path) {
    // set device
    device = torch::kCPU;
    if (torch::cuda::is_available()) {
        device = torch::kCUDA;
    }

    try {
        model = torch::jit::load(path);
        model.to(device);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return;
    }
    return;
}

std::vector<BoundingBox> YOLO::predict(const cv::Mat& image) {
    
}