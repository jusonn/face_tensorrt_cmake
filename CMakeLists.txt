cmake_minimum_required(VERSION 3.15)
project(face_tensorrt LANGUAGES CXX CUDA)

set(TARGET_NAME face_tensorrt)


#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

#set(_TensorRT_SEARCHES)

#if(TensorRT_ROOT)
#    set(_TensorRT_SEARCH_ROOT PATHS ${TensorRT_ROOT} NO_DEFAULT_PATH)
#    list(APPEND _TensorRT_SEARCHES _TensorRT_SEARCH_ROOT)
#endif()
set(PLUGINS_NEEDED OFF)
set(SAMPLE_PARSERS "none")
set(CMAKE_CXX_STANDARD 17 )
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(CUDA_INSTALL_DIR /usr/local/cuda)


if(BUILD_PLUGINS)
    list(APPEND DEPS_LIST nvinfer_plugin)
endif()
if(BUILD_PARSERS)
    list(APPEND DEPS_LIST nvuffparser nvcaffeparser nvonnxparser)
endif()


# TODO: Remove the NVINTERNAL branch once the CMake system is fully integrated into P4
if (NVINTERNAL)
    set(ONNX_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/parsers${NVINTERNAL_SUFFIX}/onnx CACHE STRING "Onnx include directory")
else()
    set(ONNX_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/parsers${NVINTERNAL_SUFFIX}/onnx CACHE STRING "Onnx include directory")
endif()


#target_include_directories(face_tensorrt
#PUBLIC /root/optimization/face_tensorrt/include
#PUBLIC ${ONNX_INCLUDE_DIR}
#PUBLIC ${CUDA_INSTALL_DIR}/include
#PRIVATE /root/optimization/face_tensorrt/include/common
##        PRIVATE ${TARGET_DIR}
#)
#set(CMAKE_CXX_STANDARD 14)


#LIST(APPEND CMAKE_MODULE_PATH "CMakeModules")
#if(WIN32)
#    set(JDK)



#set(OpenCV_STATIC ON)
#set(OpenCV_DIR ${OPENCV_HOME}/skd/native/jni)
#set(OpenCV_DIR "/usr/local/share/opencv4")
#set(OpenMP_DIR "/usr/local/share/opencv4")

#find_library(NVINFER NAMES libnvinfer.so)
#find_library(NVPARSERS NAMES nvparsers)
#find_library(NVONNXPARSERS NAMES nvonnxparser)

find_package(OpenCV REQUIRED)
#set(OpenCV_STATIC ON)

find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDA 10.0 REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${TensorRT_INCLUDE_DIRS})
if(OpenCV_FOUND)
    message("Found OpenCV")
    include_directories(${OpenCV_INCLUDE_DIR})
    message(STATUS ${OpenCV_INCLUDE_DIRS})
else()
    message("OpenCV not found, woot")
endif(OpenCV_FOUND)
if(OpenMP_FOUND)
    message("Found OpenMP")
    else()
    message("OpenMP not found, woot")
endif(OpenMP_FOUND)
if(Eigen3_FOUND)
    message("Found Eigen3")
else()
    message("Eigen3 not found, woot")
endif(Eigen3_FOUND)
if(CUDA_FOUND)
    message("Found CUDA")
    message(CUDA_INCLUDE_DIR ${CUDA_INCLUDE_DIR})
    message(CUDA_LIBRARIES ${CUDA_LIBRARIES})
else()
    message("CUDA not found, woot")
endif(CUDA_FOUND)

set( NAME_HEADERS
        include/ocrCommon.h
        include/checkBlacklist.h
#        include/blackListDetection.h
        ${OpenCV_INCLUDE_DIR}
        include/common/logger.h
#        include/arcface.h
#        include/mtcnn.h
#        include/util.h
#        include/base.h
        include/printInfo.h
#        /root/optimization/TensorRT-7.0.0.11/include/NvInfer.h
        )

set( NAME_LIBS
#        compareBlacklist.cpp
#        libOCR.cpp
#        blackListDetection.cpp
#        arcface.cpp
#        mtcnn.cpp
#        util.cpp
#        base.cpp
        src/checkBlackList.cpp
        src/face_tensorrt.cpp
        src/common/getOptions.cpp
        src/common/logger.cpp
#        src/common/sampleEngines.cpp
        src/common/sampleInference.cpp
        src/common/sampleOptions.cpp
        src/common/sampleReporting.cpp

#        src/blacklistDetection.cpp
        )

set(TRT_DIR /root/optimization/TensorRT-7.0.0.11/lib/)

include_directories("/root/optimization/face_tensorrt/include/")
include_directories("/root/optimization/face_tensorrt/include/common/")
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories("/usr/local/include/")
include_directories("/root/optimization/TensorRT-7.0.0.11/include/")
include_directories("/root/optimization/TensorRT-7.0.0.11/samples/common/")
include_directories("/usr/local/cuda/include/")
#INCLUDE_DIRECTORIES(/utils/ncnn/build/src)
#link_directories(/root/optimization/TensorRT-7.0.0.11/lib/)

set( NAME_TRT_LIBS
        "/root/optimization/TensorRT-7.0.0.11/lib/libmyelin.so"
        "/root/optimization/TensorRT-7.0.0.11/lib/libnvinfer.so"
        "/root/optimization/TensorRT-7.0.0.11/lib/libnvinfer_plugin.so"
        "/root/optimization/TensorRT-7.0.0.11/lib/libnvonnxparser.so"
        "/root/optimization/TensorRT-7.0.0.11/lib/libnvparsers.so"
#        "/usr/local/cuda/lib64/libcublas.so"
#        "/usr/local/cuda/lib64/libcublas.so.10.0"
#        "/usr/local/cuda/lib64/libcublas.so.10.0.130"
        "/usr/local/cuda/lib64/libcudart.so"
        "/usr/local/cuda/lib64/libnvrtc.so"
#        "/root/optimization/TensorRT-7.0.0.11/lib/libonnx_proto.so"
#        "/root/optimization/TensorRT-7.0.0.11/lib/libprotobuf.so"
        )



set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
set(TRT_TOOLKIT_ROOT_DIR /root/optimization/TensorRT-7.0.0.11)

find_library(MYELIN_LIB myelin HINTS
        ${TRT_TOOLKIT_ROOT_DIR}/lib NO_DEFAULT_PATH)


find_library(CUBLAS_LIB cublas HINTS
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64 NO_DEFAULT_PATH)

find_library(CUDART_LIB cudart HINTS
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64 NO_DEFAULT_PATH)

#find_library(CUBLAS_LIB cublas HINTS
#        ${CUDA_TOOLKIT_ROOT_DIR}/lib64 NO_DEFAULT_PATH)

message("CUBLAS_LIB" ${CUBLAS_LIB})
message("CUDART_LIB" ${CUDART_LIB})
message("MYELIN_LIB" ${MYELIN_LIB})
message("CUDNN_LIB" ${CUDNN_LIB})
message("CMAKE_DL_LIBS" ${CMAKE_DL_LIBS})

set(SAMPLE_DEP_LIBS
        ${CUDART_LIB}
        ${CUBLAS_LIB}
        ${CUDNN_LIB}
#        nvinfer
#        nvonnxparser
        ${RT_LIB}
        ${CMAKE_DL_LIBS}
        ${CMAKE_THREAD_LIBS_INIT}
        )
#set_target_properties(face_tensorrt PROPERTIES LINK_FLAGS "-Wl,--exclude-libs,ALL")

add_executable(face_tensorrt ${NAME_HEADERS} ${NAME_LIBS})

#FIND_LIBRARY(SAMPLE_LIBS NAMES myelin PATHS /usr/local/cuda/lib64/)
target_link_libraries(face_tensorrt ${OpenCV_LIBS} ${OpenMP_LIBS} ${CUDA_LIBRARIES} ${NAME_TRT_LIBS} pthread ${CUBLAS_LIB}
        ${MYELIN_LIB} -Wl,--unresolved-symbols=ignore-in-shared-libs)
#        /usr/local/cuda/lib64/libcudart_static.a)

