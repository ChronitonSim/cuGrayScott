#pragma once
#include <string>
#include <string_view>
#include <format>
#include <source_location>
#include <stdexcept>
#include <cuda_runtime.h>
#include <fstream>
#include <vector>

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

inline void writeBinaryFrame(
    const std::vector<float>& data, 
    int step,
    std::string_view outDir
) {
    // Generate a padded filename (e.g. out/frame_0100.bin).
    // Dissecting {:04d}:
    // : indicates the start of the formatting rules
    // 0 pads empty spaces with zeros instead of default spaces
    // 4 sets the minimum width of the output to four characters
    // d specifies that the input is a base 10 decimal integer.
    // When sorting the files alphabetically, this ensures that 
    // frame_0002.bin correctly comes before frame_0010.bin.
    std::string filename = std::format("{}/frame_{:06d}.bin", outDir, step);

    // Open file in binary mode, and overwrite (truncate) if it exists.
    // Uses bitmasking (bitwise OR) to obtain a combined flag for both
    // binary and truncation modes.
    std::ofstream out(filename, std::ios::binary | std::ios::trunc);
    
    if (!out.is_open())
        throw std::runtime_error("Failed to open file for writing: " + filename);

    // Write the raw bytes directly from the vector's underlying array.
    // Reinterpret our float data buffer as a buffr of raw bytes,
    // of conventional type (const char*).
    out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    out.close();
}