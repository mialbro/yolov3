#include <torch/cuda.h>

#include "YOLO.h"
#include "BoundingBox.h"


YOLO::YOLO(const std::string& path) {
    // set device
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

std::vector<BoundingBox> YOLO::predict(const cv::Mat& img) {
    img.convertTo(img, CV_32F, 1.0f / 255.0f);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    torch::Tensor input_tensor = torch::from_blob(img.data, {1, img.rows, img.cols, img.channels()}, device);
    input_tensor = input_tensor.permute({0, 3, 1, 2});
    
    // preprocess
    input_tensor[0][0] = input_tensor[0][0].sub(0.485).div(0.229);
    input_tensor[0][1] = input_tensor[0][1].sub(0.456).div(0.224);
    input_tensor[0][2] = input_tensor[0][2].sub(0.406).div(0.225);
    input_tensor = input_tensor.to(device);

    // forward prop
    auto result = model.forward({input_tensor}).toTensor();
    auto result_data = result.accessor<float, 2>();

    // create bounding boxes from detections
    std::vector<BoundingBox> bboxes(4);
    for (int i = 0; i < result.size(0); i++) {
        torch::Tensor tensor = torch::zeros(result_data[i].size(0));
        for (int j = 0; j < result_data[i].size(0); j++)
            tensor[j] = result_data[i][i];
        bboxes[i] = BoundingBox(tensor, img);
    }
    return bboxes;
}