# Contributing to cuGrayScott

## Commit Message Format
Each commit message consists of a **type**, a **scope**, and a **subject**:

`<type>(<scope>): <subject>`

Example: `feat(kernel): add naive 5-point stencil FDM implementation`

### 1. Types
Must be one of the following:
* **feat**: A new feature (e.g., adding a new boundary condition type).
* **fix**: A bug fix (e.g., fixing an out-of-bounds memory access).
* **perf**: A code change that improves performance (e.g., optimizing memory coalescing, using shared memory).
* **refactor**: A code change that neither fixes a bug nor adds a feature (e.g., renaming variables for clarity).
* **docs**: Documentation only changes.
* **build**: Changes that affect the build system or external dependencies (CMake).
* **style**: Changes that do not affect the meaning of the code (white-space, formatting, etc.).
* **chore**: Regular maintenance, updating `.gitignore`, etc.

### 2. Scopes
Must be one of the following, reflecting the hybrid nature of the project:
* **host**: CPU-side C++ code (e.g., `main.cpp`, host memory allocation).
* **device**: GPU memory management and transfer (e.g., `cudaMalloc`, `cudaMemcpy`).
* **kernel**: Actual CUDA device execution code (`__global__` or `__device__` functions).
* **io**: File reading, writing, and data extraction.
* **build**: CMake setup and compiler flags.
* **core**: Global parameters, physics constants, or mathematical definitions.

### 3. Subject
* Use the imperative, present tense: "change" not "changed" nor "changes".
* Do not capitalize the first letter.
* No dot (.) at the end.