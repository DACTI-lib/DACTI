set(CMAKE_EXPORT_COMPILE_COMMANDS)

#############
# fmt, spdlog
#############
include(FetchContent)
FetchContent_Declare(fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG master
    )
FetchContent_MakeAvailable(fmt)

set(GITHUB_URL_ROOT https://github.com/)
FetchContent_Declare(spdlog URL ${GITHUB_URL_ROOT}/gabime/spdlog/archive/v1.12.0.tar.gz)
FetchContent_MakeAvailable(spdlog)

#############
# tracy
#############
if(ENABLE_TRACY_PROFILING)
	FetchContent_Declare ( tracy
			GIT_REPOSITORY https://github.com/wolfpld/tracy.git
			GIT_TAG v0.11.1 #Make sure to use the same version as your local client
			GIT_SHALLOW TRUE
			GIT_PROGRESS TRUE
	)
	FetchContent_MakeAvailable(tracy)
endif()


#############
# vtu11
#############
FetchContent_Declare(
    vtu11
    GIT_REPOSITORY https://github.com/phmkopp/vtu11
    GIT_TAG        origin/master
)
FetchContent_GetProperties(vtu11)

if(NOT vtu11_POPULATED)
	FetchContent_Populate(vtu11)
  	add_subdirectory(${vtu11_SOURCE_DIR} ${vtu11_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()


#############
# opengl
#############
if(ENABLE_OPENGL_SUPPORT)
    FetchContent_Declare(
            glfw
            GIT_REPOSITORY https://github.com/glfw/glfw
    )

    FetchContent_GetProperties(glfw)
    if(NOT glfw_POPULATED)
    	FetchContent_Populate(glfw)
    	set(GLFW_BUILD_EXAMPLES OFF CACHE INTERNAL "Build the GLFW example programs")
    	set(GLFW_BUILD_TESTS OFF CACHE INTERNAL "Build the GLFW test programs")
    	set(GLFW_BUILD_DOCS OFF CACHE INTERNAL "Build the GLFW documentation")
    	set(GLFW_INSTALL OFF CACHE INTERNAL "Generate installation target")
    	set(GLFW_BUILD_WAYLAND ON CACHE INTERNAL "Build Wayland support")
    	set(GLFW_BUILD_X11 OFF CACHE INTERNAL "Build X11 support")

    	add_subdirectory(${glfw_SOURCE_DIR} ${glfw_BINARY_DIR})

        target_compile_definitions(glfw PUBLIC POSIX_REQUIRED_STANDARD=199309L)
        target_compile_definitions(glfw PUBLIC _POSIX_C_SOURCE=POSIX_REQUIRED_STANDARD)
        target_compile_definitions(glfw PUBLIC _POSIX_SOURCE=POSIX_REQUIRED_STANDARD)
        target_compile_definitions(glfw PUBLIC O_CLOEXEC)
    endif()

    FetchContent_Declare(
        glew
        GIT_REPOSITORY https://github.com/Perlmint/glew-cmake.git
        GIT_TAG        origin/master
    )
    FetchContent_GetProperties(glew)
    if(NOT glew_POPULATED)
        FetchContent_Populate(glew)
        add_subdirectory(${glew_SOURCE_DIR} ${glew_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()
endif()


#############
# libigl
#############
include(FetchContent)
FetchContent_Declare(
    libigl
    GIT_REPOSITORY https://github.com/libigl/libigl.git
    GIT_TAG v2.5.0
)
FetchContent_MakeAvailable(libigl)
igl_include(glfw)
igl_include(embree)


#############
# toml
#############
include(FetchContent)
FetchContent_Declare(
        tomlplusplus
        GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
        GIT_TAG v3.4.0
)
FetchContent_MakeAvailable(tomlplusplus)