cmake_minimum_required(VERSION 3.4)
project(gmmloc)

find_package(catkin_simple REQUIRED)
catkin_simple()

add_definitions("${CMAKE_CXX_FLAGS} -Wall -msse3 -std=c++17 -O3 -march=native -Wno-deprecated-declarations")

set(PROTO_DEFNS proto/gmmloc/GMM.proto)
set(PROTO_LIBRARIES "")
find_package(protobuf_catkin QUIET)
if (protobuf_catkin_FOUND)
  message(STATUS "Using protobuf_catkin")
  PROTOBUF_CATKIN_GENERATE_CPP(PROTO_SRCS PROTO_HDRS ${PROTO_DEFNS})
  set(PROTO_LIBRARIES ${protobuf_catkin_LIBRARIES})
else()
  message(STATUS "Using system protobuf")
  find_package(Protobuf REQUIRED)
  PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS ${PROTO_DEFNS})
  set(PROTO_LIBRARIES ${PROTOBUF_LIBRARIES})
endif()

find_package(OpenCV 3.0 REQUIRED)
# find_package(fmt)
# find_package(PCL REQUIRED COMPONENTS io)

include_directories(
	${PROJECT_SOURCE_DIR}/include
)

cs_add_library(${PROJECT_NAME}_proto
  ${PROTO_SRCS}
)
set_target_properties(${PROJECT_NAME}_proto PROPERTIES PUBLIC_HEADER "${PROTO_HDRS}")
target_include_directories(${PROJECT_NAME}_proto
                           PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(${PROJECT_NAME}_proto ${PROTO_LIBRARIES})

cs_add_library(${PROJECT_NAME} SHARED

src/gmmloc.cpp
src/gmmloc_opt.cpp

src/global.cpp
src/config.cpp

src/modules/tracking.cpp
src/modules/tracking_opt.cpp
src/modules/localization.cpp
src/modules/localization_opt.cpp

src/gmm/factors.cpp
src/gmm/gaussian.cpp
src/gmm/gaussian_mixture.cpp
src/gmm/gmm_utils.cpp

src/cv/orb_extractor.cpp
src/cv/orb_vocabulary.cpp
src/cv/orb_matcher.cpp
src/cv/pinhole_camera.cpp

src/types/frame.cpp
src/types/keyframe.cpp
src/types/map.cpp
src/types/mappoint.cpp

src/utils/dataloader.cpp
src/utils/protobuf_utils.cpp
src/utils/timing.cpp
src/utils/math_utils.cpp
src/utils/cv_utils.cpp

src/visualization/campose_visualizer.cpp
src/visualization/gmm_visualizer.cpp
src/visualization/visualizer.cpp
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${PROJECT_NAME}_proto
)

cs_add_executable(gmmloc_node node/gmmloc_node.cpp)
target_link_libraries(gmmloc_node ${PROJECT_NAME})

install(TARGETS ${PROJECT_NAME}_proto
        LIBRARY DESTINATION "lib"
        PUBLIC_HEADER DESTINATION "include"
)
cs_install()
cs_export(
	DEPENDS OpenCV
)
