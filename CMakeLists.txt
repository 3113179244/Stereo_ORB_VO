cmake_minimum_required(VERSION 3.10)
project(Stereo_ORB_VO)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 设置 Release 优化编译
set(CMAKE_BUILD_TYPE "Release")
add_compile_options(-O3 -march=native -w)

# 添加现代 OpenGL 扩展支持宏，避免 Pangolin glsl 报错
add_definitions(-DHAVE_GLEW -DGL_GLEXT_PROTOTYPES)

# 查找依赖库
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(Pangolin REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(g2o REQUIRED)
find_package(fmt REQUIRED)
find_package(DBoW3 REQUIRED)

include_directories(
  src
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
  ${Pangolin_INCLUDE_DIRS}
  ${OPENGL_INCLUDE_DIR}
  ${GLEW_INCLUDE_DIRS}
  ${DBoW3_INCLUDE_DIRS}
)

# 1. 编译核心算法库
add_library(${PROJECT_NAME}_lib SHARED
  src/Camera.cpp
  src/Config.cpp
  src/ORBmatcher.cpp
  src/ORBextractor.cpp
  src/Frame.cpp
  src/KeyFrame.cpp
  src/Map.cpp
  src/MapPoint.cpp
  src/Optimizer.cpp
  src/System.cpp
  src/MotionOnlyBA.cpp
  src/Tracker.cpp
  src/Viewer.cpp
  src/FrameDrawer.cpp
  src/KeyFrameDatabase.cpp
  src/LocalMapping.cpp
  src/LoopClosing.cpp
)

target_link_libraries(${PROJECT_NAME}_lib
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
  Sophus::Sophus
  ${Pangolin_LIBRARIES}
  ${OPENGL_LIBRARIES}
  ${GLEW_LIBRARIES}
  ${DBoW3_LIBS}
  g2o_core                  
  g2o_stuff
  g2o_types_sba
  g2o_solver_eigen
  fmt::fmt
  pthread
)

# 2. 编译 KITTI 可执行文件
add_executable(stereo_kitti src/stereo_kitti.cpp)
target_link_libraries(stereo_kitti ${PROJECT_NAME}_lib)

# 3. 编译 EuRoC 可执行文件
add_executable(stereo_euroc src/stereo_euroc.cpp)
target_link_libraries(stereo_euroc ${PROJECT_NAME}_lib)