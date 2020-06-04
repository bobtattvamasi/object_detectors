#include <time.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <queue>
#include <thread>

#include <opencv2/opencv.hpp>

#include "class_detector.h"

int main() {
  std::queue<cv::Mat> queue;
  std::condition_variable cond;
  std::mutex mut;
  std::atomic_bool stopped{false};
  std::vector<std::thread> threads;

  Detector detector;
  Config config;

  config.file_model_cfg = "configs/yolov3.cfg";
  config.file_model_weights = "configs/yolov3.weights";
  config.calibration_image_list_file_txt = "";
  config.precison = Config::FP32;
  config.log_level = INFO;
  config.detect_thresh = 0.9f;
  config.net_type = Config::YOLOV3;
  config.max_workspace_size = 1 << 30;

  std::string path = "rtsp://212.80.86.68:554/live/ch00_0";
  std::string value =
      "video_codec;h264_cuvid|fflags;nobuffer|flags;low_latency|framedrop|framerate;25|rtsp_"
      "transport;tcp";

  setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", value.c_str(), true);
  detector.Init(config);

  std::vector<Result> res;
  cv::VideoCapture vr;

  if (!vr.open(path, cv::CAP_FFMPEG))
    return 1;

  res.reserve(20);
  cv::namedWindow("image");

  char msg[64];

  std::thread consumer([&] {
    std::unique_lock<std::mutex> lock(mut);

    while (!stopped) {
      cond.wait(lock, [&] { return (!queue.empty() || stopped); });

      if (stopped)
        break;

      if (!queue.empty()) {
        auto mat_image = queue.front();
        queue.pop();
        lock.unlock();

        auto start = std::chrono::steady_clock::now();
        detector.Detect(mat_image, res);
        auto end = std::chrono::steady_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        auto fps = 1000 / duration;

        for (const auto& r : res)
          cv::rectangle(mat_image, r.rect, cv::Scalar(255, 0, 0), 2);

        std::sprintf(msg, "INFERENCE TIME/FPS: %lu ms/%lu fps", duration, fps);

        /* clang-format off */
        cv::putText(mat_image, msg,
                    cv::Point(2, 50),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 0, 0), 2, cv::FILLED);
        /* clang-format on */

        cv::imshow("image", mat_image);

        char k = cv::waitKey(5);
        if (k == 27)
          stopped = true;

        lock.lock();
      }
    }
  });

  std::thread producer([&] {
    while (!stopped) {
      cv::Mat mat_image;
      bool grabbed = vr.read(mat_image);

      if (!grabbed)
        break;

      {
        std::unique_lock<std::mutex> lock(mut);
        if (queue.size() >= 4)
          queue.pop();
        queue.push(mat_image);
      }

      cond.notify_one();
    }
  });

  threads.push_back(std::move(consumer));
  threads.push_back(std::move(producer));

  for (auto& thr : threads)
    thr.join();

  cv::destroyAllWindows();
  return 0;
}
