#pragma once

// Forward declaration of the kernel launcher.
void runGrayScottStep(
    float* d_U,
    float* d_V,
    float* d_next_U,
    float* d_next_V,
    int width,
    int height,
    float Du,
    float Dv,
    float F,
    float k,
    float dt,
    float h
);