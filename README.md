Перед тем как запускать - собери либу pydetector для cpp_trt внутри modules/cpp_trt_object_detector<br/>

mkdir build<br/>
cd build<br/>
cmake ..<br/>
make -j$(nproc)<br/>

Веса:<br/>

1) [tracker](https://drive.google.com/open?id=1GQ3mxpgVqEsoIpqUvxsJFnRxVN2lWBpG) (папка: modules_helper/deep_sort_tracker_helper/model_data/)<br/>
2) [tensorflow object detector](https://yadi.sk/d/3l04stY62OpdYQ) (папку darknet_weights положить в : modules/tf_object_detector/data)<br/>
3) Что бы скачать cfg веса для cpp_trt нужно выполнить скрипт: modules/cpp_trt_object_detector/scripts/download.py<br/>
Таким образом скачаются веса йолки в modules/cpp_trt_object_detector/configs<br/>
4) [onnx-yolov3-416](https://drive.google.com/file/d/1qvPZAF1WDG_FwGIk5WTfxDME8_Qx4qpJ/view?usp=sharing)(папка: modules/trt_object_detector/from_onnx_to_tensorrt)<br/>

Usage:

python main.py -n <one type of object detector: [tf, trt, cpp_trt]>
