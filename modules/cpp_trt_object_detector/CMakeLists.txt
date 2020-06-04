cmake_minimum_required(VERSION 3.10)
project(yolov3-trt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_INSTALL_PREFIX "/usr/local/yolo-trt" CACHE STRING "")

set(YOTRT_BUILD_PYTHON ON CACHE BOOL "")
set(YOTRT_BUILD_EXAMPLE OFF CACHE BOOL "")
set(YOTRT_PYTHON_VERSION "3.6" CACHE STRING "")

if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0)
  message(FATAL_ERROR "please use GCC 7 and higher")
elseif(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
  set(STD_LIB_WITH_FS "stdc++fs")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CXX_SOURCES
  modules/class_detector.cpp
  modules/logger.cc
  modules/trt_utils.cpp
  modules/plugin_factory.cpp
  modules/yolo_config_parser.cpp
  modules/yolo.cpp
  modules/yolov2.cpp
  modules/yolov3.cpp
  modules/yoloplugin_lib.cpp
  modules/calibrator.cpp
  modules/ds_image.cpp)

set(CUDA_SOURCES
  modules/kernel.cu)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
  -Wno-write-strings \
  -Wno-unused-variable \
  -Wno-unused-function \
  -Wno-unused-but-set-variable \
  -Wno-deprecated-declarations")

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,$ORIGIN")
set(TensorRT_ROOT /usr/local/TensorRT)

include(cmake/cuda_stuff.cmake)
include(cmake/find_tensorrt.cmake)

find_package(OpenCV REQUIRED COMPONENTS core dnn highgui imgcodecs imgproc)
find_package(Threads REQUIRED)
find_Package(gflags REQUIRED)

if(NOT gflags_FOUND)
 message(FATAL_ERROR "gflags not found")
endif()

# generate detector lib
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
   set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif("${isSystemDir}" STREQUAL "-1")

set(COMMON_LIBS
  ${CUDA_LIBRARIES}
  ${OpenCV_LIBS}
  ${TensorRT_LIBRARY}
  ${STD_LIB_WITH_FS}
  gflags)

add_library(detector SHARED ${CXX_SOURCES} ${CUDA_SOURCES})
target_link_libraries(detector PRIVATE ${COMMON_LIBS})
target_include_directories(detector PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/modules
  ${CMAKE_CURRENT_SOURCE_DIR}/extra
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${TensorRT_INCLUDE_DIRS})

target_compile_definitions(detector PRIVATE -DAPI_EXPORTS)
set_target_properties(detector PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET "hidden")

install (TARGETS detector LIBRARY DESTINATION lib)

# generate detector python module
if(YOTRT_BUILD_PYTHON)
  find_package(PythonLibs EXACT ${YOTRT_PYTHON_VERSION})
  find_package(PythonInterp EXACT ${YOTRT_PYTHON_VERSION})

  add_subdirectory(pybind11)
  set(PYBIND11_CPP_STANDARD -std=c++14)

  pybind11_add_module(pydetector MODULE THIN_LTO ${CMAKE_CURRENT_SOURCE_DIR}/wrapper/pydetector.cc)

  target_link_libraries(pydetector PRIVATE ${COMMON_LIBS} detector)
  install (TARGETS pydetector LIBRARY DESTINATION lib)
endif()

# generate detector sample app
if(YOTRT_BUILD_EXAMPLE)
  add_executable(yolov3-trt samples/sample_detector.cpp)
  target_include_directories(yolov3-trt PUBLIC ${OpenCV_INCLUDE_DIRS} modules extra)
  target_link_libraries(yolov3-trt detector ${OpenCV_LIBS} Threads::Threads)
  install (TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Build example app: ${YOTRT_BUILD_EXAMPLE}")
message(STATUS "Build python ${YOTRT_PYTHON_VERSION} module: ${YOTRT_BUILD_PYTHON}")
message(STATUS "Install path: ${CMAKE_INSTALL_PREFIX}")

set(QT_IGNORE ${QT_QMAKE_EXECUTABLE})
