# - Try to find DynaSLAM
# Set alternative paths to search for using DynaSLAM_DIR
# Once done this will define
#  DynaSLAM_FOUND - System has DynaSLAM
#  DynaSLAM_INCLUDE_DIRS - The DynaSLAM include directories
#  DynaSLAM_LIBRARIES - The libraries needed to use DynaSLAM
#  DynaSLAM_DEFINITIONS - Compiler switches required for using DynaSLAM

# TODO: This need to find dependencies properly, I can't find an example of how to do that
#find_package(OpenCV REQUIRED)
#find_package(Eigen3 REQUIRED)
#find_package(Pangolin REQUIRED)

set(_DynaSLAM_SEARCHES /usr/local)
if (DynaSLAM_DIR)
    set(_DynaSLAM_SEARCHES ${DynaSLAM_DIR} ${_DynaSLAM_SEARCHES})
endif()
find_path(DynaSLAM_INCLUDE_DIR DynaSLAM/System.h
          PATHS ${_DynaSLAM_SEARCHES} PATH_SUFFIXES include)

find_library(DynaSLAM_LIBRARY NAMES DynaSLAM libDynaSLAM
             PATHS ${_DynaSLAM_SEARCHES} PATH_SUFFIXES lib)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set DynaSLAM_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(DynaSLAM  DEFAULT_MSG
                                  DynaSLAM_LIBRARY DynaSLAM_INCLUDE_DIR)

mark_as_advanced(DynaSLAM_INCLUDE_DIR DynaSLAM_LIBRARY )

set(DynaSLAM_LIBRARIES ${DynaSLAM_LIBRARY})
set(DynaSLAM_INCLUDE_DIRS ${DynaSLAM_INCLUDE_DIR})

