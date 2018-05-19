set(EXECUTABLE opencl_solver)

file(READ param.h PARAMS)
file(READ input.cl INPUT)
configure_file(_kernel.h.in ${CMAKE_CURRENT_SOURCE_DIR}/_kernel.h)

file(GLOB SRC_LIST
	opencl_solver.cpp
	blake2b.cpp)

file(GLOB HEADER_LIST
	opencl_solver.h
	blake2b.h)

find_package(OpenCL REQUIRED)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${OPENCL_INCLUDE_DIRS})

ADD_LIBRARY(${EXECUTABLE} STATIC ${SRC_LIST} ${HEADER_LIST})
TARGET_LINK_LIBRARIES(${EXECUTABLE} ${OPENCL_LIBRARIES})
