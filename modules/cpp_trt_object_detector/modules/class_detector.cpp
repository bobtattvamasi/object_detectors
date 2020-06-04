#include "class_detector.h"

#include "class_yolo_detector.hpp"
#include "logger.h"

class Detector::Impl {
 public:
  Impl() = default;
  ~Impl() = default;

  YoloDectector _detector;
};

Detector::Detector() { impl_ = new Impl(); }
Detector::~Detector() {
  if (impl_ != nullptr)
    delete impl_;
}

void Detector::Init(const Config& config) {
  omv::Logger::Instance().Init(config.log_level);
  impl_->_detector.init(config);
}

void Detector::Detect(const cv::Mat& mat_image, std::vector<Result>& result) {
  impl_->_detector.detect(mat_image, result);
}

std::vector<Result> Detector::Detect2(const cv::Mat& mat_image) {
  std::vector<Result> vec_result;
  impl_->_detector.detect(mat_image, vec_result);
  return vec_result;
}
