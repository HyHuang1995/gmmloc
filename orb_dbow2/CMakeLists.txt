cmake_minimum_required(VERSION 2.8)
project(orb_dbow2)

# Set optimized building:
if(MSVC_IDE)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
else(MSVC_IDE)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3 -march=native")
endif(MSVC_IDE)

set(HDRS_DBOW2
  include/dbow2/BowVector.h
  include/dbow2/FORB.h 
  include/dbow2/FClass.h       
  include/dbow2/FeatureVector.h
  include/dbow2/ScoringObject.h   
  include/dbow2/TemplatedVocabulary.h)
set(SRCS_DBOW2
  src/dbow2/BowVector.cpp
  src/dbow2/FORB.cpp      
  src/dbow2/FeatureVector.cpp
  src/dbow2/ScoringObject.cpp)

set(HDRS_DUTILS
  include/dutils/Random.h
  include/dutils/Timestamp.h)
set(SRCS_DUTILS
  src/dutils/Random.cpp
  src/dutils/Timestamp.cpp)

find_package(catkin_simple REQUIRED)
catkin_simple()

find_package(OpenCV REQUIRED)

# catkin_package(
#    INCLUDE_DIRS include
#    LIBRARIES ${PROJECT_NAME}
#    CATKIN_DEPENDS 
#    DEPENDS OpenCV
#  )

# set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

include_directories(
  include
  ${catkin_INCLUDE_DIRS} 
  ${OpenCV_INCLUDE_DIRS}
)

cs_add_library(${PROJECT_NAME} SHARED 
${SRCS_DBOW2} 
${SRCS_DUTILS}
)
target_link_libraries(${PROJECT_NAME} 
${OpenCV_LIBS} 
${catkin_INCLUDE_DIRS}
)

cs_install()
cs_export()
