find_package(CUDAToolkit REQUIRED)

add_executable(exlusive_scan_intro exlusive_scan_intro.cu)
set_target_properties(exlusive_scan_intro PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(exlusive_scan_intro PRIVATE
        --extended-lambda
        -O3
)
target_include_directories(exlusive_scan_intro PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

add_executable(exclusive_scan_matmul exclusive_scan_matmul.cu)
set_target_properties(exclusive_scan_matmul PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(exclusive_scan_matmul PRIVATE
        --extended-lambda
        -O3
)
target_include_directories(exclusive_scan_matmul PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

add_executable(fibonacci fibonacci.cu)
set_target_properties(fibonacci PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(fibonacci PRIVATE
        --extended-lambda
        -O3
)
target_include_directories(fibonacci PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

add_executable(fibonacci_large fibonacci_large.cu)
set_target_properties(fibonacci_large PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(fibonacci_large PRIVATE
        --extended-lambda
        -O3
)
target_include_directories(fibonacci_large PRIVATE ${CUDAToolkit_INCLUDE_DIRS})