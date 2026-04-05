#pragma once

// Public API for the host to launch the Gray-Scott simulation step.
void runGrayScottStep(
    const float* d_U,
    const float* d_V,
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