cmake_minimum_required (VERSION 2.8.9)

project(Test)

# SET(CMAKE_CXX_FLAGS "-std=c++11 -stdlib=libc++")
set(GLUT_glut_LIBRARY /home/lz/.conan/data/cuda/10.0.130/ar/stable/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/cuda_toolkit/samples/common/inc/GL/freeglut.h)
set(GLUT_INCLUDE_DIR /home/lz/.conan/data/cuda/10.0.130/ar/stable/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/cuda_toolkit/samples/common/inc/GL/)

find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})

find_package(GLUT REQUIRED)
if (GLUT_FOUND)
    message(STATUS "Debug (lei): Libraries for glut found")
else()
    message(STATUS "Debug (lei): Failed to find glut.")
endif()

include_directories(${GLUT_INCLUDE_DIR})

add_executable(Test src/test.cpp)

target_link_libraries(Test ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})