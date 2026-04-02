#include <iostream>
#include <vector>
#include <stdexcept>
#include <algorithm>

// Grid dimensions
constexpr int N_x {256};
constexpr int N_y {256};

// Kinetic parameters
// These specific values produce a "Coral" Turing pattern
constexpr float D_u {0.16f};
constexpr float D_v {0.08f};
constexpr float F {0.035f};
constexpr float k {0.065f};

// Discretization parameters
constexpr float h {1.0f};  // Spatial step dx = dy
constexpr float dt {1.0f}; // Time step

// Row-major linearization of grid nodes.
// The x dim is indexed by i, so i is the column index;
// The y dim is indexed by j, so j is the row index.
// Thus, a node of grid coordinates (j, i) has 
// linearized coordinate k = j * N_x + i
inline int getIndex(int col, int row, int width){
    return row * width + col;
}

int main() {
    std::cout << "Starting cuGrayScott Initialization...\n";

    // Stability analysis check
    float max_D {std::max(D_u, D_v)};
    float dt_limit { (h * h)/(4.0f * max_D) };

    if (dt > dt_limit)
        throw std::runtime_error("Numerical Instability Warning: dt exceeds the von Neumann stability limit.");

    std::cout << "Stability check passed. dt = " << dt << " is <= dt_limit = " << dt_limit << "\n";

    // Total number of grid points.
    const int numNodes {N_x * N_y};

    // Host vectors for U and V concentrations.
    // Flattened 1D arrays for contiguous memory mapping.
    std::vector<float> h_U(numNodes, 1.0f);
    std::vector<float> h_V(numNodes, 0.0f);

    // Initialize a central square to seed the reaction.
    int centerX {N_x / 2};  // Floor division.
    int centerY {N_y / 2};
    int squareSize {20};

    for (int y {centerY - squareSize / 2}; y < centerY + squareSize / 2; ++y) {
        for (int x {centerX - squareSize / 2}; x < centerX + squareSize / 2; ++x) {
            int index = getIndex(x, y, N_x);
            h_U[index] = 0.5f;  // Deplete U.
            h_V[index] = 0.25f; // Inject V.
        }
    }

    std::cout << "Host memory allocated and initialized successfully.\n";
    std::cout << "Grid size: " << N_x << " x " << N_y << " points.\n";

    return 0;

}



