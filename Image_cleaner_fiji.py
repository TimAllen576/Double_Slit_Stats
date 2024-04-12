"""Removes a known bad pixel and
separates out images with only one active pixel
"""

import imagej
import numpy as np
import h5py

# Start ImageJ
ij = imagej.init("C:/Users/twlln/Progs/Fiji.app")

# Open HDF5 file
file_path = "C:/Users/twlln/Downloads/example-femm-3d.h5"
hdf5_file = h5py.File(file_path, "r")

# Get the dataset containing images
images_dataset = hdf5_file["images"]

# Select the first frame to calculate mask
first_frame = images_dataset[0]
ij.py.show(first_frame)

# Apply a median filter to smooth the image
ij.py.run_macro("""
        run("Median...", "radius=2");
        setAutoThreshold("Default dark");
        run("Convert to Mask");
        run("Fill Holes");
        run("Options...", "iterations=1 count=1 black edm=Overwrite");
        run("Invert");
        """, interactive=False)

# Save the mask
mask = ij.py.get_image()
mask_path = "path/to/save/mask.tif"
ij.py.save_as(mask, "Tiff", mask_path)

# Process each frame
for i in range(1, len(images_dataset)):
    frame = images_dataset[i]

    # Apply the previously calculated mask
    mask = ij.py.open(mask_path)
    ij.py.run("Multiply...", "value=255")

    # Apply the mask to original image
    frame = np.array(frame) * np.array(mask)

    # Save the processed image
    processed_image_path = f"path/to/save/processed_image_{i}.tif"
    ij.py.save_as(frame, "Tiff", processed_image_path)

    # Close the mask
    ij.py.close()

# Close the HDF5 file
hdf5_file.close()

# Dispose ImageJ
ij.dispose()
