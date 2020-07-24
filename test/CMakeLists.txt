CMAKE_MINIMUM_REQUIRED(VERSION 3.1)


CUDA_ADD_EXECUTABLE(add add.cu)
SET_PROPERTY(TARGET add PROPERTY FOLDER test)



CUDA_ADD_EXECUTABLE(helloWorld helloWorld.cu)
SET_PROPERTY(TARGET helloWorld PROPERTY FOLDER test)

include_directories(${OpenCV_INCLUDE_DIRS})
message(${OpenCV_INCLUDE_DIRS})

include_directories(.)

CUDA_ADD_EXECUTABLE(gaussianblur gaussianblur.cu)

TARGET_LINK_LIBRARIES(gaussianblur
 ${OpenCV_LIBS})

SET_PROPERTY(TARGET gaussianblur PROPERTY FOLDER test)

