#pragma once
#include <cuda_runtime.h>


// Row-major linearization of grid nodes.
// The x dim is indexed by i, so i is the column index;
// The y dim is indexed by j, so j is the row index.
// Thus, a node of grid coordinates (j, i) has 
// linearized coordinate k = j * N_x + i
inline int getIndex(int col, int row, int width){
    return row * width + col;
}

__device__
inline int d_getIndex(int col, int row, int width){
    return row * width + col;
}


