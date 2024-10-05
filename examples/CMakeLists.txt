add_executable(cuSZp_gpu_f32_api cuSZp_gpu_f32_api.cpp)
add_executable(cuSZp_cpu_f32_api cuSZp_cpu_f32_api.cpp)
add_executable(cuSZp_gpu_f64_api cuSZp_gpu_f64_api.cpp)
add_executable(cuSZp_cpu_f64_api cuSZp_cpu_f64_api.cpp)

target_link_libraries(cuSZp_gpu_f32_api szp_cu_compat)
target_link_libraries(cuSZp_cpu_f32_api szp_cu_compat)
target_link_libraries(cuSZp_gpu_f64_api szp_cu_compat)
target_link_libraries(cuSZp_cpu_f64_api szp_cu_compat)

# Installation of executables
install(TARGETS 
    cuSZp_gpu_f32_api 
    cuSZp_cpu_f32_api 
    cuSZp_gpu_f64_api 
    cuSZp_cpu_f64_api
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)