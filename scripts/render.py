import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import os

# --- PARAMETERS ---
N_X, N_Y = 4096, 4096
FPS = 30
MP4_FILENAME = "gray_scott_coral.mp4"

def load_frame(filename):
    # np.fromfile is the fastest way to read raw C++ binary dumps
    return np.fromfile(filename, dtype=np.float32).reshape((N_Y, N_X))

def process_directory(target_dir):
    output_file = os.path.join(target_dir, MP4_FILENAME)
    
    # 1. Pipeline Check: Does the video already exist?
    if os.path.exists(output_file):
        print(f"[SKIP] '{output_file}' already exists. Moving to next directory...")
        return

    # 2. Grab all binary files in this specific directory
    search_pattern = os.path.join(target_dir, 'frame_*.bin')
    files = glob.glob(search_pattern)
    
    if not files:
        print(f"[SKIP] No .bin files found in '{target_dir}'.")
        return

    # Sort numerically (robust against different path lengths)
    files.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))

    print(f"[{target_dir}] Found {len(files)} frames. Initializing renderer...")

    # 3. Setup the figure
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='black')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # Remove all margins
    ax.axis('off') # Hide axes

    initial_data = load_frame(files[0])
    im = ax.imshow(initial_data, cmap='inferno', vmin=0.0, vmax=0.35, animated=True)

    def update(frame_idx):
        data = load_frame(files[frame_idx])
        im.set_array(data)
        
        if frame_idx % 50 == 0: # Reduced print frequency for cleaner logs
            print(f"[{target_dir}] Rendering frame {frame_idx} / {len(files)}")
            
        return [im]

    print(f"[{target_dir}] Encoding MP4... (This might take a minute)")
    
    ani = animation.FuncAnimation(fig, update, frames=len(files), blit=True)
    ani.save(output_file, fps=FPS, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    
    # 4. CRITICAL: Free memory before moving to the next directory!
    plt.close(fig)
    print(f"[SUCCESS] Animation saved to {output_file}\n")

def main():
    # Find all directories named "out_something"
    # This handles both running from the root dir ('out_*') or scripts dir ('../out_*')
    base_dirs = glob.glob('out_*') + glob.glob('../out_*')
    target_dirs = [d for d in base_dirs if os.path.isdir(d)]
    
    if not target_dirs:
        print("Error: Could not find any directories starting with 'out_'.")
        return

    print(f"Pipeline started. Found target directories: {target_dirs}\n")
    
    # Run the pipeline on every directory
    for directory in target_dirs:
        process_directory(directory)
        
    print("Pipeline complete.")

if __name__ == '__main__':
    main()