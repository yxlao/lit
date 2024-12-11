cmake_minimum_required(VERSION 3.10)
project("DeepSDF")

set(OpenGL_GL_PREFERENCE "GLVND")
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(ZLIB REQUIRED)

include(ExternalProject)

# Global config
set(CMAKE_CXX_STANDARD 17)

# Define the source files
set(SOURCES
    src/ShaderProgram.cpp
    src/Utils.cpp
    src/cnpy.cpp
)

# CLI11
set(clt11_GIT_REPOSITORY "https://github.com/CLIUtils/CLI11.git")
set(clt11_GIT_TAG "v2.1.1")
ExternalProject_Add(clt11
    GIT_REPOSITORY ${clt11_GIT_REPOSITORY}
    GIT_TAG        ${clt11_GIT_TAG}
    CMAKE_ARGS     -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/CLI11
    TIMEOUT 10
)
add_library(cli11_ifc INTERFACE)
add_dependencies(cli11_ifc clt11)
target_include_directories(cli11_ifc INTERFACE ${CMAKE_BINARY_DIR}/CLI11/include)

# Eigen
set(eigen_GIT_REPOSITORY "https://gitlab.com/libeigen/eigen.git")
set(eigen_GIT_TAG "3.4.0")
ExternalProject_Add(eigen_ext
    GIT_REPOSITORY ${eigen_GIT_REPOSITORY}
    GIT_TAG ${eigen_GIT_TAG}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_DISCONNECTED TRUE
    TIMEOUT 10
)
ExternalProject_Get_Property(eigen_ext SOURCE_DIR)
set(EIGEN_INCLUDE_DIR ${SOURCE_DIR})
add_library(eigen_ifc INTERFACE)
add_dependencies(eigen_ifc eigen_ext)
target_include_directories(eigen_ifc INTERFACE ${EIGEN_INCLUDE_DIR})

# Pangolin
set(pangolin_GIT_REPOSITORY "https://github.com/stevenlovegrove/Pangolin.git")
set(pangolin_GIT_TAG "v0.6")
ExternalProject_Add(pangolin_ext
    GIT_REPOSITORY ${pangolin_GIT_REPOSITORY}
    GIT_TAG ${pangolin_GIT_TAG}
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/pangolin
               -DEIGEN_INCLUDE_DIR=${EIGEN_INCLUDE_DIR}
               -DCMAKE_BUILD_TYPE=Release
               -DBUILD_EXAMPLES=OFF
               -DBUILD_PANGOLIN_PYTHON=OFF
               -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
    DEPENDS eigen_ext
    UPDATE_DISCONNECTED TRUE
    TIMEOUT 10
)
add_library(pangolin_ifc INTERFACE)
add_dependencies(pangolin_ifc pangolin_ext)
target_include_directories(pangolin_ifc INTERFACE ${CMAKE_BINARY_DIR}/pangolin/include)
target_link_libraries(pangolin_ifc INTERFACE ${CMAKE_BINARY_DIR}/pangolin/lib/libpangolin.so)

set(nanoflann_GIT_REPOSITORY "https://github.com/jlblancoc/nanoflann.git")
set(nanoflann_GIT_TAG "v1.3.2")
ExternalProject_Add(nanoflann_ext
    GIT_REPOSITORY ${nanoflann_GIT_REPOSITORY}
    GIT_TAG ${nanoflann_GIT_TAG}
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/nanoflann
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    TIMEOUT 10
)
ExternalProject_Get_Property(nanoflann_ext SOURCE_DIR)
add_library(nanoflann_ifc INTERFACE)
add_dependencies(nanoflann_ifc nanoflann_ext)
target_include_directories(nanoflann_ifc INTERFACE ${SOURCE_DIR}/include)

# Application targets
add_executable(PreprocessMesh src/PreprocessMesh.cpp ${SOURCES})
target_link_libraries(PreprocessMesh PRIVATE
    cli11_ifc
    eigen_ifc
    pangolin_ifc
    nanoflann_ifc
    GLEW::GLEW
    ${OPENGL_LIBRARIES}
    ${ZLIB_LIBRARIES}
)
target_compile_features(PreprocessMesh PRIVATE cxx_std_14)
set_target_properties(PreprocessMesh PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
)
target_include_directories(PreprocessMesh PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)

add_executable(SampleVisibleMeshSurface src/SampleVisibleMeshSurface.cpp ${SOURCES})
target_link_libraries(SampleVisibleMeshSurface PRIVATE
    cli11_ifc
    eigen_ifc
    pangolin_ifc
    nanoflann_ifc
    GLEW::GLEW
    ${OPENGL_LIBRARIES}
    ${ZLIB_LIBRARIES}
)
target_compile_features(SampleVisibleMeshSurface PRIVATE cxx_std_14)
set_target_properties(SampleVisibleMeshSurface PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
)
target_include_directories(SampleVisibleMeshSurface PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)
