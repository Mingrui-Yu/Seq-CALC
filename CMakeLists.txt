
cmake_minimum_required(VERSION 2.8)
project(deeplcd)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

# colors, because why not :)
string(ASCII 27 Esc)
set(RED "${Esc}[31m")
set(GREEN "${Esc}[32m")
set(ENDL "${Esc}[m")

MESSAGE("${GREEN}Build type: ${CMAKE_BUILD_TYPE}${ENDL}")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall -O3 -march=native ")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native -std=c++11")
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules) # FindEigen3.cmake

find_package(Boost COMPONENTS filesystem REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(CUDA QUIET)

# option(CPU_ONLY  "Build Caffe without CUDA support" ON) # TODO: rename to USE_CUDA
# add_definitions( -DCPU_ONLY=0 )

if (NOT Caffe_ROOT_DIR)
	set(Caffe_ROOT_DIR $ENV{HOME}/caffe)
endif()

MESSAGE("${GREEN}Using Caffe_ROOT_DIR: ${Caffe_ROOT_DIR}${ENDL}")
set(Caffe_LIBS ${Caffe_ROOT_DIR}/build/lib/libcaffe.so)
set(Caffe_INCLUDE_DIRS ${Caffe_ROOT_DIR}/build/install/include)

MESSAGE("Caffe_LIBS: ${Caffe_LIBS}")
MESSAGE("Caffe_INCLUDE_DIRS: ${Caffe_INCLUDE_DIRS}")



include_directories(
	${OpenCV_INCLUDE_DIR}
	${Boost_INLCLUDE_DIRS}
	${Caffe_INCLUDE_DIRS}
	${EIGEN3_INCLUDE_DIR}
	${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}
)


if (${CUDA_FOUND})
 	include_directories(${CUDA_INCLUDE_DIRS})
	MESSAGE("CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
endif()

file(GLOB SOURCES "src/${PROJECT_NAME}/[A-z]*.cpp")

add_library(${PROJECT_NAME} SHARED
	${SOURCES}
)

target_link_libraries(${PROJECT_NAME}
	${OpenCV_LIBS}
	${Caffe_LIBS}
	${Boost_LIBRARIES}
	${EIGEN3_LIBS}
	glog
	pthread
)
if (${CUDA_FOUND})
	target_link_libraries(${PROJECT_NAME}
		${CUDA_LIBRARIES}
		${CUDA_CUBLAS_LIBRARIES}
		${CUDA_curand_LIBRARY}
	)
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
	${Caffe_INCLUDE_DIRS}
)


### Default model download ###########################

message( "\n\n${GREEN}Downloading default model if not already downloaded\n\n${ENDL}" )
execute_process(
	COMMAND ./get_model.sh
	WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)


#### Demo ###########################################

# message( "\n\nCopying images and model to build directory for demo and testing....\n\n" )
# execute_process(COMMAND cp -r ../src/images .)
execute_process(COMMAND cp -r ../calc_model .)


add_executable( demo demo/demo.cpp )
target_link_libraries( demo
	${PROJECT_NAME}
	${OpenCV_LIBS}
)

add_executable( test1 demo/test1.cpp )
target_link_libraries( test1
	${PROJECT_NAME}
	${OpenCV_LIBS}
)

add_executable( test2 demo/test2.cpp )
target_link_libraries( test2
	${PROJECT_NAME}
	${OpenCV_LIBS}
)

add_executable( test3 demo/test3.cpp )
target_link_libraries( test3
	${PROJECT_NAME}
	${OpenCV_LIBS}
)

add_executable( test4 demo/test4.cpp )
target_link_libraries( test4
	${PROJECT_NAME}
	${OpenCV_LIBS}
)

add_executable( test4_2 demo/test4_2.cpp )
target_link_libraries( test4_2
	${PROJECT_NAME}
	${OpenCV_LIBS}
)

add_executable( test5 demo/test5.cpp )
target_link_libraries( test5
	${PROJECT_NAME}
	${OpenCV_LIBS}
)













