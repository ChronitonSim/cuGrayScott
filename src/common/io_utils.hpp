#pragma once

#include <string>
#include <format>
#include <source_location>
#include <stdexcept>
#include <cuda_runtime.h>

// Inline allows us to define this function in a header without violating the ODR.
inline void cudaCheck(cudaError_t err,
                      std::source_location location = std::source_location::current()) {
    
    if (err != cudaSuccess) {
        std::string msg = std::format("CUDA error at {}:{} code={} \"{}\"\n",
                                      location.file_name(),
                                      location.line(),
                                      static_cast<int>(err),
                                      cudaGetErrorString(err));
        throw std::runtime_error(msg);
    }
}