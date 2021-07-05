
cmake_minimum_required(VERSION 3.0.2)
project(yolo)

# Enable C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_BUILD_TYPE Debug)

add_compile_options(-std=c++14)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

set(OpenCV_DIR ~/libraries/opencv4.5/opencv/build/)
set(Torch_DIR ~/libraries/libtorch/share/cmake/Torch)

# find thirdparty packages
find_package(OpenCV 4.5 REQUIRED)
find_package(Torch REQUIRED)

include_directories(
  include
  ${TORCH_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}_ex
    src/YOLO.cpp
    src/BoundingBox.cpp
)

add_executable(${PROJECT_NAME}
  src/main.cpp
)


target_link_libraries(${PROJECT_NAME}
  ${TORCH_LIBRARIES}
  ${PROJECT_NAME}_ex
  ${OpenCV_LIBS}
)
