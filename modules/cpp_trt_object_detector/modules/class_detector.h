#ifndef CLASS_DETECTOR_H_
#define CLASS_DETECTOR_H_

#include <iostream>

#include <opencv2/opencv.hpp>

#include "API.h"
#include "logger.h"

struct Result {
  int id = -1;
  float prob = 0.f;
  cv::Rect rect;
};

struct Config {
  enum ModelType { YOLOV2 = 0, YOLOV3, YOLOV2_TINY, YOLOV3_TINY };
  enum Precision { INT8 = 0, FP16, FP32 };

  std::string file_model_cfg{"configs/yolov3.cfg"};
  std::string file_model_weights{"configs/yolov3.weights"};
  std::string calibration_image_list_file_txt{};

  OmvSeverity log_level{OmvSeverity::WARN};
  ModelType net_type{YOLOV3};
  Precision precison{FP32};

  std::size_t max_workspace_size{1 << 30};
  float detect_thresh{0.9};
  int gpu_id{0};
};

class API Detector {
 public:
  explicit Detector();
  Detector(int);
  ~Detector();

  void Init(const Config& config);
  void Detect(const cv::Mat& mat_image, std::vector<Result>&);
  std::vector<Result> Detect2(const cv::Mat& mat_image);

 private:
  Detector(const Detector&);
  const Detector& operator=(const Detector&);
  class Impl;
  Impl* impl_;
};

#endif  // !CLASS_QH_DETECTOR_H_
