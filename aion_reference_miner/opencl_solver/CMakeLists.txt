set(EXECUTABLE opencl_solver)

file(GLOB SRC_LIST
	opencl_solver.cpp
	blake2b.cpp)

file(GLOB HEADER_LIST
	opencl_solver.h
	blake2b.h)

find_package(OpenCL REQUIRED)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${OpenCL_INCLUDE_DIRS})

ADD_LIBRARY(${EXECUTABLE} STATIC ${SRC_LIST} ${HEADER_LIST})
TARGET_LINK_LIBRARIES(${EXECUTABLE} ${OpenCL_LIBRARIES})
