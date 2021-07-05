
typedef struct BoundingBox {
    cv::Mat image;
    int x, y, w, h;
    double prob;
    std::vector<int> class_dist;
}


class YOLO {
public:
    YOLO(const std::string&);
    std::vector<BoundingBox> predict(const cv::Mat&);
    void showObject(cont BoundingBox&);
public:
    torch::Device device;
    torch::jit::script::Module yolo;
}