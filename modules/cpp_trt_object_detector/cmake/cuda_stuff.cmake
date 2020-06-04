include(FindCUDA)

enable_language(CUDA)

message(STATUS "NVCC version: ${CMAKE_CUDA_COMPILER_VERSION}")

set(CUDA_ARCH_LIST Auto CACHE LIST
    "List of CUDA architectures (e.g. Pascal, Volta, etc) or \
compute capability versions (6.1, 7.0, etc) to generate code for. \
Set to Auto for automatic detection (default)."
)

cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${CUDA_ARCH_LIST})

list(APPEND CUDA_NVCC_FLAGS ${CUDA_ARCH_FLAGS})

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wno-deprecated-declarations")
endif()
