#pragma once

// Create a dedicated namespace to prevent naming collisions
namespace Params {

    // Simulation grid dimensions.
    constexpr int N_x {4096};
    constexpr int N_y {4096};

    // Kinetic parameters.
    // These specific values produce a "Coral" Turing pattern.
    constexpr float D_u {0.16f};
    constexpr float D_v {0.08f};
    constexpr float F {0.035f};
    constexpr float k {0.065f};

    // Discretization parameters.
    constexpr float h {1.0f};  // Spatial step dx = dy
    constexpr float dt {1.0f}; // Time step
}