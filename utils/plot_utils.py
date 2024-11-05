import imageio
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from PIL import Image
import glob
import numpy as np
import os


def video_from_snapshots(snapshot_folder, save_path, fps=8):
    '''
    Make a video from all images in a folder and save to path.
    '''

    print(f'Making video of snapshots in {snapshot_folder} ...')

    # Ensure the save_path has a proper video file extension
    if not save_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise ValueError("Save path must have a valid video file extension (.mp4, .avi, .mov, .mkv)")

    # Initialize the writer with fps and codec for .mp4
    writer = imageio.get_writer(save_path, fps=fps)#, codec='libx264' if save_path.endswith('.mp4') else None)

    # Retrieve and sort all image paths in the snapshot folder
    image_paths = sorted(glob.glob(os.path.join(snapshot_folder, '*')))

    for im_path in image_paths:
        image = Image.open(im_path)
        
        # Convert image to numpy array and append to video
        image_array = np.array(image)
        writer.append_data(image_array)

    writer.close()
    print('Video saved!')