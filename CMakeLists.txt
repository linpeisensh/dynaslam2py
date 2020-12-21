cmake_minimum_required(VERSION 2.8)
project(DynaSLAM_PYTHON_BINDING)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})
set(CMAKE_MODULE_PATH "/usr/stud/linp/storage/user/linp/DynaSLAM_PYTHON_BINDING/cmake_modules")
LIST(APPEND CMAKE_MODULE_PATH "/usr/stud/linp/storage/user/linp/DynaSLAM_PYTHON_BINDING/cmake_modules")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

find_package(Pangolin)

# RPATH fiddling for the library, so that the installed version can find the linked libraries
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(DynaSLAM_INCLUDE_DIR "/usr/stud/linp/storage/user/linp/DynaSLAM/include;/usr/stud/linp/storage/user/linp/DynaSLAM")
set(Boost_INCLUDE_DIRS "/usr/include/boost")
set(PYTHON_INCLUDE_DIRS "/usr/include/python3.6m")
set(PYTHON_NUMPY_INCLUDE_DIR "/usr/include")

include_directories(
${PROJECT_SOURCE_DIR}/src
${DynaSLAM_INCLUDE_DIR}
${Boost_INCLUDE_DIRS}
${PYTHON_INCLUDE_DIRS}
${PYTHON_NUMPY_INCLUDE_DIR}
)
if (Pangolin_FOUND)
  include_directories(${Pangolin_INCLUDE_DIRS})
endif()

find_package(PythonLibs 3.6 REQUIRED)
find_package(NumPy)
find_package(Boost 1.65.1 REQUIRED COMPONENTS python-py36)
find_package(OpenCV REQUIRED)
find_package(DynaSLAM REQUIRED)


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

set(TARGET_MODULE_NAME orbslam2)
add_library(${TARGET_MODULE_NAME} SHARED
src/ORBSlamPython.cpp
src/pyboost_cv2_converter.cpp
src/pyboost_cv3_converter.cpp)
set_target_properties(${TARGET_MODULE_NAME} PROPERTIES PREFIX "")

target_link_libraries(${TARGET_MODULE_NAME} 
${DynaSLAM_LIBRARIES}
${OpenCV_LIBS}
${Boost_LIBRARIES}
${PYTHON_LIBRARIES}
)
if (Pangolin_FOUND)
  target_link_libraries(${TARGET_MODULE_NAME} ${Pangolin_LIBRARIES})
endif()

install(TARGETS ${TARGET_MODULE_NAME} DESTINATION lib/python3.6/)

