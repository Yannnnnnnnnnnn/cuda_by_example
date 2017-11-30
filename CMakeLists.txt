CMAKE_MINIMUM_REQUIRED(VERSION 3.1)

project(cuda_by_example)

#folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#outpath
SET(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/bin")  

#Find GLUT
find_package(GLUT REQUIRED)
include_directories(${GLUT_INCLUDE_DIRS})
link_directories(${GLUT_LIBRARY_DIRS})
if(NOT GLUT_FOUND)
    message(ERROR " GLUT not found!")
endif(NOT GLUT_FOUND)

#Find OpenGL
find_package(OpenGL REQUIRED)
include_directories(${OpenGL_INCLUDE_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
if(NOT OPENGL_FOUND)
    message(ERROR " OPENGL not found!")
endif(NOT OPENGL_FOUND)

#CUDA
find_package(CUDA)
if (CUDA_FOUND)
	add_subdirectory(appendix_a)
	add_subdirectory(chapter03)
	add_subdirectory(chapter04)
	add_subdirectory(chapter05)
	add_subdirectory(chapter06)
	add_subdirectory(chapter07)
	add_subdirectory(chapter08)
	add_subdirectory(chapter09)
	add_subdirectory(chapter10)
	add_subdirectory(chapter11)
else()
	message("CUDA not found!")
endif()
