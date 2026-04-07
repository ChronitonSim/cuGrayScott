import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import os

# --- PARAMETERS ---
N_X, N_Y = 256, 256
FPS = 30
OUTPUT_FILE = "../out/gray_scott_coral.mp4"

def load_frame(filename):
    # np.fromfile is the fastest way to read raw C++ binary dumps
    # We reshape the flat 1D array back into our 2D simulation grid
    return np.fromfile(filename, dtype=np.float32).reshape((N_Y, N_X))

def main():
    # Grab all binary files and sort them numerically
    files = glob.glob('../out/frame_*.bin')
    # Extract the number from "frame_XXXX.bin" and sort numerically
    files.sort(key=lambda f: int(f.split('_')[1].split('.')[0]))
    
    if not files:
        print("Error: No .bin files found in out/")
        return

    print(f"Found {len(files)} frames. Initializing renderer...")

    # Setup the figure (black background looks best for these patterns)
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='black')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # Remove all margins
    ax.axis('off') # Hide axes

    # Load the first frame to setup the colormap
    initial_data = load_frame(files[0])
    
    # We visualize the 'V' concentration. 
    # V typically ranges from 0.0 to roughly 0.3 in the Coral pattern.
    # 'inferno' or 'magma' are excellent colormaps for this.
    im = ax.imshow(initial_data, cmap='inferno', vmin=0.0, vmax=0.35, animated=True)

    # The update function called by FuncAnimation for every frame
    def update(frame_idx):
        data = load_frame(files[frame_idx])
        im.set_array(data)
        
        if frame_idx % 20 == 0:
            print(f"Rendering frame {frame_idx} / {len(files)}")
            
        return [im]

    print("Encoding MP4... (This might take a minute)")
    
    # blit=True optimizes rendering by only redrawing pixels that have changed
    ani = animation.FuncAnimation(fig, update, frames=len(files), blit=True)
    ani.save(OUTPUT_FILE, fps=FPS, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    
    print(f"Success! Animation saved to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()