cmake_minimum_required(VERSION 3.8)
project(Stereo_ORB_VO)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE "Debug")
add_compile_options(-Wextra -Wpedantic -w)
add_compile_options(-O3 -march=native)

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
  ${PANGOLIN_INCLUDE_DIRS}
  ${OPENGL_INCLUDE_DIR}
  ${DBow3_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}_lib
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
  pango_display        
  ${OPENGL_LIBRARIES}
  ${DBoW3_LIBS}
  g2o_core                  
  g2o_stuff
  g2o_types_sba
  g2o_solver_eigen
  GLEW::GLEW
  pthread
  fmt::fmt
)

add_executable(Stereo_ORB_VO src/main.cpp)

target_link_libraries(Stereo_ORB_VO ${PROJECT_NAME}_lib)

install(TARGETS ${PROJECT_NAME}_lib Stereo_ORB_VO
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)