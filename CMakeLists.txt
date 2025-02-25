cmake_minimum_required(VERSION 3.16)
project(DACTI)

# fix policy warning
cmake_policy(SET CMP0135 NEW)

set(ENABLE_OPENGL_SUPPORT "OFF")
set(ENABLE_TRACY_PROFILING "OFF")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fopenmp -O3 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fopenmp")

set(CMAKE_NINJA_FORCE_RESPONSE_FILE "ON" CACHE BOOL "Force Ninja to use response files.")

##########
# DACTI LIB
##########
include_directories(${PROJECT_SOURCE_DIR}/src/lib)
file(GLOB_RECURSE LIB_SRC src/lib/*.cpp)
add_library(dacti ${LIB_SRC})
target_compile_features(dacti PUBLIC cxx_std_20)

##########
# LOCAL LIBS
##########
find_package(VTK 9.3 COMPONENTS 
	CommonCore
	CommonDataModel
	FiltersGeneral
	IOXML
)
set(LIBS ${VTK_LIBRARIES})

find_package(OpenMP)
set(LIBS ${LIBS} OpenMP::OpenMP_CXX)

##########
# EXTERNAL LIBS
##########
add_subdirectory(ext)

if(ENABLE_OPENGL_SUPPORT)
	find_package(OpenGL)
	set(LIBS ${LIBS} glfw OpenGL::GL libglew_static)
	message("OpenGL visualization enabled")
endif()

if(ENABLE_TRACY_PROFILING)
	set(LIBS ${LIBS} Tracy::TracyClient)
	message("Tracy enabled")
endif()

set(LIBS ${LIBS}
	fmt::fmt
	vtu11::vtu11
	spdlog
	Eigen3::Eigen 
	igl::core 
	igl::embree 
	igl::glfw 
	tomlplusplus::tomlplusplus
)

##########
# LINK LIBS
##########
target_link_libraries(dacti ${LIBS})

##########
# EXAMPLES
##########
function(link_example NAME)
    target_link_libraries(${NAME} dacti)
    if(ENABLE_OPENGL_SUPPORT)
        target_compile_definitions(${NAME} PUBLIC GL_VIS)
    endif()
    if(ENABLE_TRACY_PROFILING)
        target_compile_definitions(${NAME} PUBLIC DACTI_PROFILER_ON)
    endif()
    target_include_directories(${NAME} PUBLIC ${INC_PATH})
    target_compile_definitions(${NAME} PUBLIC DACTI_EXAMPLE_NAME="${NAME}")

    set_target_properties(${NAME} PROPERTIES CXX_STANDARD 20)
    set_target_properties(${NAME} PROPERTIES CXX_STANDARD_REQUIRED ON)
    # set_target_properties(${NAME} PROPERTIES COMPILE_FLAGS "-fopenmp -O3 -march=native")
endfunction()

# build EXAMPLE_FILES list using glob
file(GLOB_RECURSE EXAMPLE_FILES src/examples/*.cpp)

# add example for each .cpp file in EXAMPLE_FILES
foreach(example_file ${EXAMPLE_FILES})
    get_filename_component(example_name ${example_file} NAME_WE)
    add_executable(${example_name} ${example_file})
    link_example(${example_name})
endforeach()