#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "class_detector.h"

namespace pybind11 {
namespace detail {
template <>
struct type_caster<cv::Rect> {
  PYBIND11_TYPE_CASTER(cv::Rect, _("tuple_xywh"));

  bool load(handle obj, bool) {
    if (!pybind11::isinstance<pybind11::tuple>(obj)) {
      std::logic_error("Rect should be a tuple!");
      return false;
    }
    pybind11::tuple rect = reinterpret_borrow<pybind11::tuple>(obj);
    if (rect.size() != 4) {
      std::logic_error("Rect (x,y,w,h) tuple should be size of 4");
      return false;
    }

    value = cv::Rect(rect[0].cast<int>(), rect[1].cast<int>(), rect[2].cast<int>(), rect[3].cast<int>());
    return true;
  }

  static handle cast(const cv::Rect& rect, return_value_policy, handle) {
    return pybind11::make_tuple(rect.x, rect.y, rect.width, rect.height).release();
  }
};

template <>
struct type_caster<cv::Mat> {
 public:
  PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));

  bool load(handle obj, bool) {
    array b = reinterpret_borrow<array>(obj);
    buffer_info info = b.request();

    int nh = 1;
    int nw = 1;
    int nc = 1;
    int ndims = info.ndim;
    if (ndims == 2) {
      nh = info.shape[0];
      nw = info.shape[1];
    } else if (ndims == 3) {
      nh = info.shape[0];
      nw = info.shape[1];
      nc = info.shape[2];
    } else {
      char msg[64];
      std::sprintf(msg, "Unsupported dim %d, only support 2d, or 3-d", ndims);
      throw std::logic_error(msg);
      return false;
    }

    int dtype;
    if (info.format == format_descriptor<unsigned char>::format()) {
      dtype = CV_8UC(nc);
    } else if (info.format == format_descriptor<int>::format()) {
      dtype = CV_32SC(nc);
    } else if (info.format == format_descriptor<float>::format()) {
      dtype = CV_32FC(nc);
    } else {
      throw std::logic_error("Unsupported type, only support uchar, int32, float");
      return false;
    }

    value = cv::Mat(nh, nw, dtype, info.ptr);
    return true;
  }

  static handle cast(const cv::Mat& mat, return_value_policy, handle) {
    std::string format = format_descriptor<unsigned char>::format();
    size_t elemsize = sizeof(unsigned char);
    int nw = mat.cols;
    int nh = mat.rows;
    int nc = mat.channels();
    int depth = mat.depth();
    int type = mat.type();
    int dim = (depth == type) ? 2 : 3;

    if (depth == CV_8U) {
      format = format_descriptor<unsigned char>::format();
      elemsize = sizeof(unsigned char);
    } else if (depth == CV_32S) {
      format = format_descriptor<int>::format();
      elemsize = sizeof(int);
    } else if (depth == CV_32F) {
      format = format_descriptor<float>::format();
      elemsize = sizeof(float);
    } else {
      throw std::logic_error("Unsupport type, only support uchar, int32, float");
    }

    std::vector<size_t> bufferdim;
    std::vector<size_t> strides;
    if (dim == 2) {
      bufferdim = {(size_t)nh, (size_t)nw};
      strides = {elemsize * (size_t)nw, elemsize};
    } else if (dim == 3) {
      bufferdim = {(size_t)nh, (size_t)nw, (size_t)nc};
      strides = {(size_t)elemsize * nw * nc, (size_t)elemsize * nc, (size_t)elemsize};
    }
    return array(buffer_info(mat.data, elemsize, format, dim, bufferdim, strides)).release();
  }
};

}  // namespace detail
}  // namespace pybind11

PYBIND11_MODULE(pydetector, m) {
  m.doc() = "pybind11 yolo detector trt plugin";

  /* clang-format off */
  pybind11::class_<Config> config(m, "Config");

  config.def(pybind11::init<>())
      .def_readwrite("file_model_cfg",&Config::file_model_cfg)
      .def_readwrite("file_model_weights",&Config::file_model_weights)
      .def_readwrite("calibration_image_list_file_txt",
                     &Config::calibration_image_list_file_txt)
      .def_readwrite("detect_thresh",&Config::detect_thresh)
      .def_readwrite("max_workspace_size", &Config::max_workspace_size)
      .def_readwrite("gpu_id",&Config::gpu_id)
      .def_readwrite("net_type",&Config::net_type)
      .def_readwrite("precision",&Config::precison)
      .def_readwrite("log_level", &Config::log_level);

  pybind11::enum_<OmvSeverity>(config, "LogLevel")
      .value("FATAL", OmvSeverity::FATAL)
      .value("ERROR", OmvSeverity::ERROR)
      .value("WARN", OmvSeverity::WARN)
      .value("INFO", OmvSeverity::INFO)
      .value("DEBUG", OmvSeverity::DEBUG)
      .export_values();

  pybind11::enum_<Config::ModelType>(config, "ModelType")
      .value("YOLOV2", Config::ModelType::YOLOV2)
      .value("YOLOV3", Config::ModelType::YOLOV3)
      .value("YOLOV2_TINY", Config::ModelType::YOLOV2_TINY)
      .value("YOLOV3_TINY", Config::ModelType::YOLOV3_TINY)
      .export_values();

  pybind11::enum_<Config::Precision>(config, "Precision")
      .value("INT8", Config::Precision::INT8)
      .value("FP16", Config::Precision::FP16)
      .value("FP32", Config::Precision::FP32)
      .export_values();

  pybind11::class_<Result>(m, "Result")
      .def_readwrite("id",&Result::id)
      .def_readwrite("prob",&Result::prob)
      .def_readwrite("rect",&Result::rect);

  pybind11::class_<Detector>(m, "Detector")
      .def(pybind11::init<>())
      .def("init", &Detector::Init)
      .def("detect", &Detector::Detect2,
           pybind11::return_value_policy::move);

  /* clang-format on */
}
