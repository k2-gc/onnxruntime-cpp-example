cmake_minimum_required(VERSION 3.22)
project(sample_app C CXX)
find_package(OpenCV REQUIRED)

string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra")
string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")
set(ONNXRUNTIME_ROOTDIR "/usr/local")
set(CMAKE_CXX_STANDARD 17)


add_executable(${PROJECT_NAME} 
    main.cpp
)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime"
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    "${ONNXRUNTIME_ROOTDIR}/lib/onnxruntime/libonnxruntime.so"
    ${OpenCV_LIBRARIES}
)