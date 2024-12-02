'''
This script uses Mortensen's dipole fitting code diPOLE.py from 'Supplementary Software' at the bottom of https://www.nature.com/articles/ncomms9621
I've translated it into python3 to work, and modified Estimate() so that it returns the position, orientation, and covariance matrix.
That code is for a single spot on a single frame, so here we just loop over every blob in a frame, and then over every frame.

One complication is that to do this we need to be able to identify spots in a frame. For this, we rely on thunderSTORM.
This script runs thunderSTORM on a directory of frames. We take those localisations, extract an NxN pixel region centred on them, and then apply Mortensen to that patch.
We then replace the localisations in the thunderSTORM results with these new adjusted localisations.
The output is a typical thunderSTORM results table. I tried to automatically turn this into an image with imageJ, but couldn't.

One of the main issues currently is that I've struggled to integrate ImageJ here.
It runs thunderSTORM fine, but I would prefer it to run headless if possible.
I also tried to get it to run at the end to generate the visualisation of the results, but have been hitting a wall with that.
pyimagej seems difficult to work with, but it might just be me.

Also, I don't have the right parameters for the experimental setup,
or for the initial thunderSTORM run. So correct that.
And make it read them in from somewhere.
Need to read in the pixel scale from the image metadata or something. Is that available?

Running on /mnt/cryosil_ro/AttoDRY800/Developments_Autumn_2024/2024-10-09_ASIL240923C05/StormData/StormData_1/Results
and using the parameters from protocol.txt

Oh and it's super slow. Like 15 seconds per spot / 5 minutes per frame. So this would never work for our usual 10,000 frame stack

--------------------

First: run thunderstorm on all frames. This gets the initial localisations.
Then loop over every frame i:
    Run mortensen_single_frame()
      This will consider all x,y in the thunderstorm results which have frame=i
      It will run extract_patches() on frame=i
      It will loop over all patches in that frame, run Mortensen on it
      Append the resulting (framenumber, xs, ys) to the overall array
Add the xs and ys to the existing thunderstorm results table
(because Mortensen is done relative to a small patch centred on the thunderstorm localisations, so just need to add it on)
LA: For now we are going to park the thunderstorm layer, we need to generate a simulation of dipoles with different orientations, calculate the centroids
    coordinates of those dipoles, then pass these coordinates to detect the blobs, calculate the patches, then loop over the blobs and fit. I am going to comment out all 
    thunderstorm and imageJ parts for now, we can revisit later.
'''

import diPOLE_python3
from pylab import *
import numpy as np
import cv2
import pandas as pd
import subprocess
import os
import time

# from config_params import *
#from thunderstorm_run_macro import run_thunderstorm, reconstruct
# from thunderstorm_reconstruct_macro import reconstruct

from tifffile import imread
from argparse import Namespace
import glob
import importlib.util

file_path = "/home/wgq72938/Documents/octo-cryoclem-smlm/smlm/demos/demo_simulate_image_stack.py"
spec = importlib.util.spec_from_file_location("demo_simulate_image_stack", file_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

AppClass = module.Application

class CustomApplication(AppClass):
    def simulation3(self):
        #print("Custom behavior before simulation3")
    
        # Call parent class simulation3
        result = super().simulation3()
        #print(f"simulation3 raw result: {result}, type: {type(result)}, shape: {result.shape}")
        
        # Process result based on its structure
        if isinstance(result, np.ndarray):
            if result.ndim == 2:  # Assume 2D array where each row is [x, y]
                result = [[(x, y) for x, y in result]]  # Wrap in a list (single frame)
            elif result.ndim == 1 and len(result) == 2:  # Single point case
                result = [[(result[0], result[1])]]
            else:
                raise ValueError(f"Unexpected result structure: {result}")

        #print(f"Processed simulation3 result: {result}")
        
        # Check frame count
        if len(result) < 200:
            #print("Simulation returned insufficient frames")
            tiff_file = self.wait_for_tiff_file()
            self.centroids = self.extract_centroids_from_tiff(tiff_file)
        else:
            self.centroids = result

        # Log centroids
        #print("Centroids of each dipole:")
        #for frame_idx, frame_centroids in enumerate(self.centroids):
        #    print(f"Frame {frame_idx + 1}: {frame_centroids}")

        #print("Custom behavior after simulation3")
        return self.centroids
    
    def wait_for_tiff_file(self, timeout=60):
        """
        Waits for the image_stack_fixdip.tif file to be created.
        Returns the full path to the TIFF file.
        """
        base_dir = "/home/wgq72938/Documents/Hinterer/fixed-dipole-issue/mortensen/simulation_results"
        start_time = time.time()

        while True:
            # Find the latest dataset_* subdirectory
            subdirs = [d for d in glob.glob(f"{base_dir}/dataset_*") if os.path.isdir(d)]
            if not subdirs:
                raise FileNotFoundError("No dataset directories found.")

            latest_subdir = max(subdirs, key=os.path.getmtime)
            tiff_file = os.path.join(latest_subdir, "image_stack_fixdip.tif")

            # Check if the file exists
            if os.path.exists(tiff_file):
                time.sleep(2)
                return tiff_file

            # Check for timeout
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                raise TimeoutError(f"TIFF file not found within {timeout} seconds.")

            time.sleep(1)  # Wait before checking again

    @staticmethod
    def extract_centroids_from_tiff(tiff_path, threshold=30, min_area=1):
        """
        Reads a TIFF file and extracts centroids from each frame.
        """
        # Load the TIFF file
        tiff_stack = imread(tiff_path)
        print(f"Loaded TIFF file with {tiff_stack.shape} frames.")

        all_centroids = []

        for frame_idx, frame in enumerate(tiff_stack):
            print(f"Processing frame {frame_idx + 1}...")

            # Convert the frame to 8-bit (if it's not already)
            if frame.dtype != np.uint8:
                frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Convert frame to binary using threshold
            _, binary_frame = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)

            # Detect contours/blobs
            contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Extract centroids
            centroids = []
            for contour in contours:
                if cv2.contourArea(contour) >= min_area:  # Filter by minimum area
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = M["m10"] / M["m00"]
                        cy = M["m01"] / M["m00"]
                        centroids.append((cx, cy))

            #print(f"  Found {len(centroids)} centroids in frame {frame_idx + 1}.")
            all_centroids.append(centroids)

        return all_centroids
    
# Setup arguments and instantiate the custom application class
args = Namespace(output="/home/wgq72938/Documents/Hinterer/fixed-dipole-issue/mortensen/simulation_results", other_arg="object")
app_instance = CustomApplication()
app_instance._args = args

# Run simulation3
centroids = app_instance.simulation3()

def blob_detect_all_frames(centroids, pixel_width):
    """
    Rescale and structure centroids
    """

    #print("Received centroids: ", centroids)
    centroids_image_coords = []

    for frame_idx, frame_centroids in enumerate(centroids):
        for x, y in frame_centroids:
            x_image_coord = int(x / pixel_width)
            y_image_coord = int(y / pixel_width)

            # Add (frame, x, y) tuple to list
            centroids_image_coords.append((frame_idx + 1, x_image_coord, y_image_coord))

    return centroids_image_coords

centroids_image_coords = blob_detect_all_frames(centroids, pixel_width=51)
#print(f"Processed centroids (image coordinates): {centroids_image_coords}")

'''
def extract_patches(tiff_stack, centroids_image_coords, patch_width):
    """
    Extract patches around given centroids from the image.

    Args:
        image (ndarray): The input image from which to extract the patches.
        centroids (list of tuple): A list of (x, y) coordinates of the centroids.
        patch_size (int): The size of the patch to extract (default is 12).

    Returns:
        list of ndarray: A list of extracted patches, or None for out of bounds.
    """
    all_patches = []

    for frame_idx, frame in enumerate(tiff_stack, start=1):
        patches=[]

        for centroid in centroids_image_coords:
            f, x, y = centroid

            # only operate on the current frame number
            if f == frame_idx:

                # Define the coordinates for the patch
                x_start = x - patch_width // 2
                x_end = x + patch_width // 2
                y_start = y - patch_width // 2
                y_end = y + patch_width // 2

                # # if you decide to ignore out-of-bounds patches:
                # # Check for out-of-bounds
                # if x_start >= 0 and x_end <= image.shape[1] and y_start >= 0 and y_end <= image.shape[0]:
                #     # Extract the patch
                #     patch = image[y_start:y_end, x_start:x_end]
                #     patches.append(patch)

                # if you decide to pad out-of-bounds patches:

                # Determine the padding needed for each side
                pad_left = max(0, -x_start)  # How many pixels need padding on the left
                pad_right = max(0, x_end - frame.shape[1])  # How many pixels need padding on the right
                pad_top = max(0, -y_start)  # How many pixels need padding on the top
                pad_bottom = max(0, y_end - frame.shape[0])  # How many pixels need padding on the bottom

                # Apply padding to the image
                padded_frame = np.pad(frame,
                                    ((pad_top, pad_bottom), (pad_left, pad_right)),
                                    mode='constant', constant_values=0)

                # Recalculate the new patch coordinates based on the padded image
                x_start_padded = x_start + pad_left
                x_end_padded = x_end + pad_left
                y_start_padded = y_start + pad_top
                y_end_padded = y_end + pad_top

                # Extract the patch from the padded image
                patch = padded_frame[y_start_padded:y_end_padded, x_start_padded:x_end_padded]
                patches.append(patch)

        # Append patches from this frame to the list of all patches        
        all_patches.extend(patches)

        print(f"Extracted {len(patches)} patches from frame {frame_idx}")
                
    return patches
    '''

def extract_patches(tiff_stack, centroids_image_coords, patch_width):
    """
    Extract patches around centroids from the frames in a TIFF stack.
    Args:
        tiff_stack (ndarray): 3D array where each slice is a frame.
        centroids_image_coords (list of tuple): List of (frame, x, y) coordinates.
        patch_width (int): The width of the square patch to extract.
    Returns:
        list: Extracted patches.
    """
    all_patches = []

    for frame_idx, frame in enumerate(tiff_stack, start=1):
        patches = []

        for centroid in centroids_image_coords:
            frame_number, x, y = centroid

        for centroid in centroids_image_coords:
            if not (len(centroid) == 3 and isinstance(centroid[0], int)):
                raise ValueError(f"Invalid centroid format: {centroid}")

            # Skip centroids that don't belong to the current frame
            if frame_number != frame_idx:
                continue

            # Define patch bounds
            x_start = x - patch_width // 2
            x_end = x + patch_width // 2
            y_start = y - patch_width // 2
            y_end = y + patch_width // 2

            # Calculate padding needed
            pad_left = max(0, -x_start)
            pad_right = max(0, x_end - frame.shape[1])
            pad_top = max(0, -y_start)
            pad_bottom = max(0, y_end - frame.shape[0])

            # Apply padding to the frame
            padded_frame = np.pad(
                frame,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode='constant',
                constant_values=0,
            )

            # Recalculate patch coordinates in the padded frame
            x_start_padded = max(0, x_start) + pad_left
            x_end_padded = min(frame.shape[1], x_end) + pad_left
            y_start_padded = max(0, y_start) + pad_top
            y_end_padded = min(frame.shape[0], y_end) + pad_top

            # Extract the patch
            patch = padded_frame[y_start_padded:y_end_padded, x_start_padded:x_end_padded]
            patches.append(patch)

        all_patches.extend(patches)
        print(f"Extracted {len(patches)} patches from frame {frame_idx}")

    return all_patches

# Ensure centroids_image_coords is in the correct format
print(f"Centroids Image Coordinates: {centroids_image_coords[:10]}")  # Print the first 10 for debugging

# Check if centroids_image_coords is a list of tuples with four elements (frame, x, y)
for entry in centroids_image_coords:
    if not (isinstance(entry, tuple) and len(entry) == 3):
        raise ValueError(f"Invalid centroid format: {entry}")
print("All centroids are valid")

# Load the TIFF file
try:
    tiff_file = app_instance.wait_for_tiff_file()
    tiff_stack = imread(tiff_file)
    print(f"Loaded TIFF file: {tiff_file} with {tiff_stack.shape[0]} frames.")

    # Ensure centroids are within valid frame range
    #num_frames = tiff_stack.shape[0]
    #centroids_image_coords = [
    #    (frame, x, y) for frame, x, y in centroids_image_coords if 1 <= frame <= num_frames
    #]
    #print(f"Validated Centroids: {centroids_image_coords[:10]}")

    # Ensure centroids are within frame bounds

    print(f"Centroids before filtering: {centroids_image_coords[:10]}")  # Debug output

    num_frames = tiff_stack.shape[0]
    centroids_image_coords = [
        (frame, x, y) for frame, x, y in centroids_image_coords if 1 <= frame <= num_frames
    ]
    print(f"Filtered centroids (valid frames): {centroids_image_coords[:10]}")  # Debug output

    # Extract patches
    patch_width = 12
    current_frame = tiff_stack[0]
    all_patches = extract_patches(tiff_stack, centroids_image_coords, patch_width)
    print(f"Total patches extracted: {len(all_patches)}")

    # Extract patches for the first frame as a test
    #patch_width = 12
    #current_frame = tiff_stack[0]  # First frame
    #all_patches = extract_patches(tiff_stack, centroids_image_coords, patch_width)

    #print(f"Total patches extracted {len(all_patches)}")

except FileNotFoundError as e:
    print(f"Error: {e}")
except ValueError as e:
    print(f"Invalid centroids: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


########## FIT ############
def mortensen_single_frame(image,
                           current_frame_number,
                           centroids_image_coords,
                           patch_width,
                           peak_emission_wavelength,
                           pixel_width,
                           magnification,
                           numerical_aperture,
                           ref_ind_immersion,
                           ref_ind_imaging,
                           ref_ind_buffer,
                           initvals,
                           initpix,
                           deltapix,
                           Sfloor,
                           inverse_gain,
                           sigma_noise):
    """
    Perform Mortensen estimation on a single frame given the extracted centroids.
    """
    # Subtract the background roughly
    image = np.clip(image - np.mean(image), 0, 255).astype(np.uint8)

    # Extract 12x12 patches around each centroid in the current frame
    blob_patches = extract_patches(image, centroids_image_coords, patch_width)

    # Create instance of MLEwT (for each frame)
    track = diPOLE_python3.MLEwT(peak_emission_wavelength,
                                 pixel_width,
                                 magnification,
                                 numerical_aperture,
                                 ref_ind_immersion,
                                 ref_ind_imaging,
                                 ref_ind_buffer,
                                 initvals,
                                 initpix,
                                 deltapix,
                                 Sfloor,
                                 inverse_gain,
                                 sigma_noise)

    # Loop over all blobs
    x_list, y_list, phi_list, theta_list, covariance_list = [], [], [], [], []

    for i, blob in enumerate(blob_patches, 1):
        start_blob = time.time()
        print(f"Analysing blob {i}/{len(blob_patches)}")

        # Perform estimation
        x_est, y_est, theta_est, phi_est, cov_mat = track.Estimate(blob)
        x_list.append(x_est)
        y_list.append(y_est)
        theta_list.append(theta_est)
        phi_list.append(phi_est)
        covariance_list.append(cov_mat)

        end_blob = time.time()
        elapsed_time_blob = end_blob - start_blob
        print(f"Time: {elapsed_time_blob:.4f} seconds on this blob")

    return x_list, y_list, theta_list, phi_list, covariance_list

# --------------------
# Mortensen run params
# --------------------
# Experimental parameters
peak_emission_wavelength = 500.0 # Peak emission wavelength
ref_ind_immersion = 1.52 # RI of immersion oil/optics
ref_ind_imaging = 1.0 # Refractive index of imaging medium (typically that of air, i.e. np = 1.0)
ref_ind_buffer = 1.31 # RI of buffer
numerical_aperture = 2.17 # Numerical aperture of objective
magnification = 215.0 # Magnification of composite microscope
pixel_width = 51.0 # Pixel width (nm per px)

# PSF parameters
photon_number = 10000.0 # Photon number?
background_level = 1.0 # Background level?
mu = 0.1 # Probe location?
nu = 0.1 # ...
phi = 2 * np.pi / 3.0 # inclination
theta = 0.5 # azimuthal angle
deltaz = -30.0 # Distance from design focal plane

# EMCCD parameters
inverse_gain = 1./100.
sigma_noise = 2 #12.6
Sfloor = 300.0
gain = 1.0 / inverse_gain

patch_width = 12 # size of NxN pixel patch around blob centroid to consider
# --------------------

# Initial thunderstorm run to get blob location

# Load data
tiff_file = app_instance.wait_for_tiff_file()
tiff_stack = imread(tiff_file)
print(tiff_stack)
frame_paths = [tiff_stack[i] for i in range(tiff_stack.shape[0])]
num_frames = tiff_stack.shape[0]

# Directory where results will be stored
results_dir = '/home/wgq72938/Documents/Hinterer/dipole-issue/mortensen-loop/simulation_results'

# Generate results filename with a timestamp
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
results_filename = f"Fit_results_{current_time}.csv"

# Combine the directory and filename to create the full results path
results_path = os.path.join(results_dir, results_filename)

# Ensure the results directory exists, if not, create it
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

print(f"Results will be saved to: {results_path}")

# find centroids using gaussian fitting thunderstorm
centroids_image_coords = blob_detect_all_frames(centroids, pixel_width)

# Initial guess params
initvals = array([mu, nu, background_level, photon_number, phi, theta, deltaz]) # initial PSF values
deltapix = patch_width / 2 # centre of patch around blob
initpix = (deltapix, deltapix) # centre of patch around blob

# Mortensen run on each blob in each frame
x_ests, y_ests, theta_ests, phi_ests, covariance_ests = [], [], [], [], []
for i in range(tiff_stack.shape[0]):
    start_frame = time.time() # record the start time for this frame

    image = tiff_stack[i]

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(f"Processing frame {i + 1}/{tiff_stack.shape[0]}, image shape: {image.shape}")

    single_frame_results = list(mortensen_single_frame(image=image,
                               current_frame_number=i+1, 
                               centroids_image_coords=centroids_image_coords, 
                               patch_width=patch_width,
                               peak_emission_wavelength=peak_emission_wavelength,
                               pixel_width=pixel_width,
                               magnification=magnification,
                               numerical_aperture=numerical_aperture,
                               ref_ind_immersion=ref_ind_immersion,
                               ref_ind_imaging=ref_ind_imaging,
                               ref_ind_buffer=ref_ind_buffer,
                               initvals=initvals,
                               initpix=initpix,
                               deltapix=deltapix,
                               Sfloor=Sfloor,
                               inverse_gain=inverse_gain,
                               sigma_noise=sigma_noise,))

    x_ests.append(single_frame_results[0])
    y_ests.append(single_frame_results[1])
    theta_ests.append(single_frame_results[2])
    phi_ests.append(single_frame_results[3])
    covariance_ests.append(single_frame_results[4])

    end_frame = time.time()
    elapsed_time_frame = end_frame - start_frame
    elapsed_time_frame = elapsed_time_frame/60
    print(f"Time: {elapsed_time_frame:.4f} minutes on this frame")

# make sure list is flat, because it needs to be for results table
x_ests = [item for sublist in x_ests for item in sublist]
y_ests = [item for sublist in y_ests for item in sublist]
theta_ests = [item for sublist in theta_ests for item in sublist]
phi_ests = [item for sublist in phi_ests for item in sublist]
covariance_ests = [item for sublist in covariance_ests for item in sublist]

print(len(x_ests))
print(len(y_ests))
print(len(theta_ests))
print(len(phi_ests))
print(len(covariance_ests))

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Ensure the results file exists before reading
if not os.path.exists(results_path):
    print(f"Creating a dummy results file at: {results_path}")
    dummy_df = pd.DataFrame({"x [nm]": [], "y [nm]": []})  # Replace with actual structure
    dummy_df.to_csv(results_path, index=False)

# Attempt to read the file
try:
    print(f"Reading results file: {results_path}")
    df = pd.read_csv(results_path)
except FileNotFoundError as e:
    print(f"Error: {e}")
    raise

# generate a thunderstorm-style results table with new x,y localisations
#mortensen_results_path = '/home/wgq72938/Documents/Hinterer/dipole-issue/mortensen-loop/simulation_results/mortensen_results.csv'
#output_img_path = '/home/wgq72938/Documents/Hinterer/dipole-issue/mortensen-loop/reconstruction.png'

df = pd.read_csv(results_path)
if len(x_ests) != len(df) or len(y_ests) != len(df):
    raise ValueError("The length of the new x and y arrays must match the number of rows in the CSV file.")
df['x [nm]'] += x_ests
df['y [nm]'] += y_ests
df.to_csv(results_path, index=False)

# --------------------
# generating image from results table
# [not working - go and do it in fiji manually for now]
# --------------------
# # this doesn't work due to some conflict or something
# reconstruct(mortensen_results_path, output_img_path)
# # try getting round it by running the macro via terminal
# # !!! this has input/output paths hard-coded !!!
# command = "/home/tfq96423/fiji-linux64/Fiji.app/ImageJ-linux64 -macro /home/tfq96423/Documents/cryoCLEM/dipole-issue/mortensen-loop/reconstruct.ijm"
# subprocess.run(command, shell=True, check=True)
#
# # show in napari
# command2 = f"napari {output_img_path}"
# subprocess.run(command2, shell=True, check=True)
