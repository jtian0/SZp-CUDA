cmake_minimum_required(VERSION 3.18)
project(cuSZp
        LANGUAGES CUDA CXX C
        VERSION 0.0.2
        DESCRIPTION "Error-bounded GPU lossy compression library"
)

# general cmake setup
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(BUILD_SHARED_LIBS "prefer shared libaries" ON)

find_package(CUDAToolkit REQUIRED)

include(GNUInstallDirs)
include(CTest)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/szp_version.h.in
        ${CMAKE_CURRENT_BINARY_DIR}/include/szp_version.h)
add_library(szp_cu_compile_settings INTERFACE)

target_compile_definitions(
        szp_cu_compile_settings
        INTERFACE $<$<COMPILE_LANG_AND_ID:CUDA,Clang>:__STRICT_ANSI__>)
target_compile_options(
        szp_cu_compile_settings
        INTERFACE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda
        --expt-relaxed-constexpr -Wno-deprecated-declarations>)
target_compile_features(szp_cu_compile_settings INTERFACE cxx_std_17 cuda_std_17)

target_include_directories(
        szp_cu_compile_settings
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# create a namespaced ALIAS for the compile settings, etc
add_library(SZP::cu_compile_settings ALIAS szp_cu_compile_settings)

add_library(szp_cu_compat
        src/cuSZp_f32.cu
        src/cuSZp_f64.cu
        src/cuSZp_utility.cu
        src/cuSZp_timer.cu
        src/cuSZp_entry_f32.cu
        src/cuSZp_entry_f64.cu
)
target_link_libraries(szp_cu_compat
        PUBLIC SZP::cu_compile_settings CUDA::cudart
)

add_library(szp_cu_module src/szp_module.cu)
add_library(szp_cu_driver src/szp_driver.cc)
add_library(szp_cu_demo src/szp_demo.cc)

target_link_libraries(szp_cu_module
        PUBLIC SZP::cu_compile_settings
        CUDA::cudart
)
target_link_libraries(szp_cu_driver
        PUBLIC SZP::cu_compile_settings
        szp_cu_module CUDA::cudart
)
target_link_libraries(szp_cu_demo
        PUBLIC SZP::cu_compile_settings
        szp_cu_module szp_cu_driver CUDA::cudart
)

add_executable(szpcli src/szp_cli.cc)
set_source_files_properties(src/szp_cli.cc PROPERTIES LANGUAGE CUDA)
target_link_libraries(szpcli PRIVATE szp_cu_driver szp_cu_module szp_cu_demo szp_cu_compat)
set_target_properties(szpcli PROPERTIES OUTPUT_NAME szp)

option(SZP_BUILD_EXAMPLES "Option to enable building example programs" ON)

if(SZP_BUILD_EXAMPLES)
        add_subdirectory(examples)
endif()

# installation
install(TARGETS
        szp_cu_compile_settings
        szp_cu_compat
        szp_cu_module
        szp_cu_driver
        EXPORT SZPTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# install the executable
install(TARGETS szpcli
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# export the targets to a script
install(EXPORT SZPTargets
        FILE SZPTargets.cmake
        NAMESPACE SZP::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SZP
)

# generate and install package configuration files
include(CMakePackageConfigHelpers)

write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/SZPConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/SZPConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/SZPConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SZP
)

install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/SZPConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/SZPConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SZP
)