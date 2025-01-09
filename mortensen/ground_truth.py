import cv2
import os
import time
import numpy as np
from pylab import *
import pandas as pd
from argparse import Namespace
import glob
import importlib.util
import matplotlib.pyplot as plt
from argparse import Namespace
import diPOLE_python3
from tifffile import imread
from datetime import datetime

def ground_truth_with_drawing(image_path, output_path, grid_size=(7, 7), spacing=10):
    """
    Compute the ground truth centroids of a grid of dipoles and draw them on the image.

    Parameters:
        image_path (str): Path to the fixed image containing the dipoles.
        output_path (str): Path to save the image with drawn centroids.
        grid_size (tuple): Number of rows and columns in the dipole grid.
        spacing (int): Pixel distance between adjacent dipoles in the grid.

    Returns:
        centroids (list): List of (x, y) tuples representing the centroid locations.
    """
    # Load the fixed image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Determine the center of the image
    height, width = image.shape
    center_x, center_y = width // 2, height // 2

    # Generate grid of centroids around the center
    centroids = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            x = center_x + (j - (grid_size[1] - 1) // 2) * spacing
            y = (center_y + (i - (grid_size[0] - 1) // 2) * spacing) - 2
            centroids.append((x, y))

            # Draw a red dot at the centroid location
            cv2.circle(image, (x, y), radius=1, color=(255, 0, 0), thickness=-1)

    # Save the image with drawn centroids
    cv2.imwrite(output_path, image)
    print(f"Image with centroids saved to {output_path}")

    return centroids

def find_centroids(image_path, output_path, threshold=15, min_area=4):
    """
    Automatically detect centroids of bright regions (dipoles) in the image.
    Parameters:
        image_path (str): Path to the fixed image containing the dipoles.
        output_path (str): Path to save the image with drawn centroids.
        threshold (int): Pixel intensity threshold for binarization.
        min_area (int): Minimum area of detected regions to consider as dipoles.
    Returns:
        centroids (list): List of (x, y) tuples representing the detected centroid locations.
    """
    # Load the fixed image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Threshold the image to create a binary mask
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            # Calculate the centroid of the contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))

                # Draw the centroid on the image
                cv2.circle(image, (cx, cy), radius=1, color=(0, 255, 0), thickness=-1)

    # Save the image with detected centroids
    cv2.imwrite(output_path, image)
    print(f"Image with detected centroids saved to {output_path}")

    return centroids

def compare_centroids(image_path, ground_truth, detected, output_path):
    """
    Draw both ground truth and detected centroids on the same image for comparison.

    Parameters:
        image_path (str): Path to the fixed image containing the dipoles.
        ground_truth (list): Ground truth centroids as a list of (x, y) tuples.
        detected (list): Detected centroids as a list of (x, y) tuples.
        output_path (str): Path to save the image with drawn centroids.
    """
    # Load the fixed image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Convert to color image for better visualization
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw ground truth centroids (red)
    for x, y in ground_truth:
        cv2.circle(image, (x, y), radius=1, color=(0, 0, 255), thickness=-1)

    # Draw corrected detected centroids (green)
    for x, y in detected:
        cv2.circle(image, (x, y), radius=1, color=(0, 255, 0), thickness=-1)

    # Save the comparison image
    cv2.imwrite(output_path, image)
    print(f"Comparison image saved to {output_path}")

def draw_patches(image_path, centroids, output_path, patch_width=12): 
    """ 
    Draw patches around the given centroids on the image and print their coordinates. 
    Parameters: image_path (str): Path to the image file. centroids (list): List of (x, y) tuples representing the centroids. 
    output_path (str): Path to save the image with patches drawn. 
    patch_width (int): Width of the square patches. 
    Returns: patch_coordinates (list): List of coordinates for each patch. 
    """ 
    # Load the image 
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 
    if image is None: 
        raise FileNotFoundError(f"Image not found at {image_path}") 
    
    # Convert to color image for better visualization 
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) 

    patch_coordinates = [] 
    for cx, cy in centroids: 
        # Define the patch coordinates 
        x_start = max(0, cx - patch_width // 2) 
        x_end = min(image.shape[1], cx + patch_width // 2) 
        y_start = max(0, cy - patch_width // 2) 
        y_end = min(image.shape[0], cy + patch_width // 2) 

        # Store the patch coordinates 
        patch_coordinates.append((x_start, y_start, x_end, y_end)) 

        # Draw a rectangle around the patch 
        cv2.rectangle(image, (x_start, y_start), (x_end, y_end), color=(255, 255, 0), thickness=1) 

    # Save the image with patches 
    cv2.imwrite(output_path, image) 
    print(f"Image with patches saved to {output_path}") 
    #print(f"Patch coordinates: {patch_coordinates}") 
    return patch_coordinates
    

########## FIT ############
#def mortensen_single_frame(image,
#                           current_frame_number,
                        #    centroids_image_coords,
                        #    patch_width,
                        #    peak_emission_wavelength,
                        #    pixel_width,
                        #    magnification,
                        #    numerical_aperture,
                        #    ref_ind_immersion,
                        #    ref_ind_imaging,
                        #    ref_ind_buffer,
                        #    initvals,
                        #    initpix,
                        #    deltapix,
                        #    Sfloor,
                        #    inverse_gain,
                        #    sigma_noise):
    # """
    # Perform Mortensen estimation on a single frame given the extracted centroids.
    # """
    # # Subtract the background roughly
    # image = np.clip(image - np.mean(image), 0, 255).astype(np.uint8)

    # # Validate centroids (skip those that are too close to the edges for a full patch)
    # valid_centroids = []
    # for cx, cy in centroids_image_coords:
    #     # Ensure centroid is within bounds such that the patch doesn't go out of the image.
    #     if patch_width // 2 <= cx < image.shape[1] - patch_width // 2 and \
    #        patch_width // 2 <= cy < image.shape[0] - patch_width // 2:
    #         valid_centroids.append((cx, cy))  # Store as (x, y)

    # if len(valid_centroids) == 0:
    #     return [], [], [], [], []

    # # Extract patches around each valid centroid
    # blob_patches = []
    # for cx, cy in valid_centroids:
    #     # Define the patch coordinates
    #     x_start = cx - patch_width // 2
    #     x_end = cx + patch_width // 2
    #     y_start = cy - patch_width // 2
    #     y_end = cy + patch_width // 2

    #     # Extract the patch and append it to blob_patches
    #     patch = image[y_start:y_end, x_start:x_end]
    #     if patch.shape == (patch_width, patch_width):
    #         blob_patches.append(patch)
    #     else:
    #         print(f"Skipped patch extraction at ({cx}, {cy}) due to invalid shape {patch.shape}")

    # if len(blob_patches) == 0:
    #     return [], [], [], [], []

    # # Create an instance of MLEwT
    # track = diPOLE_python3.MLEwT(
    #     peak_emission_wavelength,
    #     pixel_width,
    #     magnification,
    #     numerical_aperture,
    #     ref_ind_immersion,
    #     ref_ind_imaging,
    #     ref_ind_buffer,
    #     initvals,
    #     initpix,
    #     deltapix,
    #     Sfloor,
    #     inverse_gain,
    #     sigma_noise
    # )

    # # Process each patch
    # x_list, y_list, theta_list, phi_list, covariance_list = [], [], [], [], []
    # for i, blob in enumerate(blob_patches, 1):
    #     try:
    #         x_est, y_est, theta_est, phi_est, cov_mat = track.Estimate(blob)
    #         x_list.append(x_est)
    #         y_list.append(y_est)
    #         theta_list.append(theta_est)
    #         phi_list.append(phi_est)
    #         covariance_list.append(cov_mat)
    #         print(f"Blob {i}: x={x_est}, y={y_est}, theta={theta_est}, phi={phi_est}")
    #     except Exception as e:
    #         print(f"Error processing blob {i}: {e}")

    # return x_list, y_list, theta_list, phi_list, covariance_list

def mortensen_single_frame(image, 
                           current_frame_number,
                           dipole_centroids,
                           patch_width,
                           peak_emission_wavelength,
                           pixel_width,
                           magnification,
                           numerical_aperture,
                           ref_ind_imaging,
                           ref_ind_buffer,
                           initvals,
                           initpix,
                           deltapix,
                           Sfloor,
                           inverse_gain,
                           sigma_noise):
    """
    Perform Mortensen estimation on a single frame given the extracted centroids
    """
    # Subtract the background roughly
    image = np.clip(image - np.mean(image), 0, 255).astype(np.uint8)

    # Validate centroids (skip those that are too close to the edges for a full patch)
    valid_centroids = []
    for frame, cx, cy in dipole_centroids:
        # Ensure centroid is within the bounds such that the patch doesn't go out of the image
        if patch_width // 2 <= cx < image.shape[1] - patch_width // 2 and \
           patch_width // 2 <= cy < image.shape[0] - patch_width // 2:
            valid_centroids.append((frame, cx, cy))
    
    if len(valid_centroids) == 0:
        print(f"No valid centroids in frame {current_frame_number}")
        return [], [], [], [], []
    
    # Extract patches around each valid centroid
    blob_patches = [
        image[cy - patch_width // 2 : cy + patch_width // 2,
              cx - patch_width // 2 : cx + patch_width //2]
        for _, cx, cy in valid_centroids
    ]

    # Check patch dimensions
    for i, patch in enumerate(blob_patches, 1):
        if patch.shape != (patch_width, patch_width):
            print(f"Skipping patch {i}: invalid shape {patch.shape}")
            continue

    # Create instance of MLEwT
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
    
    # Process each patch
    x_list, y_list, theta_list, phi_list, covariance_list = [], [], [], [], []
    for i, blob in enumerate(blob_patches, 1):
        try:
            x_ests, y_ests, theta_ests, phi_ests, cov_mat = track.Estimate(blob)
            x_list.append(x_ests)
            y_list.append(y_ests)
            theta_list.append(theta_ests)
            phi_list.append(phi_ests)
            covariance_list.append(cov_mat)
            print(f"Blob {i}: x={x_ests}, y={y_ests}, theta={theta_ests}, phi={phi_ests}")
        except Exception as e:
            print(f"Error processing blob {i}: {e}")

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
pixel_width = 25.6 # Pixel width (nm per px)

# PSF parameters
photon_number = 1000000.0 # Photon number?
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
tiff_stack = cv2.imread("/home/wgq72938/Documents/Hinterer/fixed-dipole-issue/mortensen/image_stack_fixdip_maxproject.png") #questo deve essere l'imm dei dipoli

# Directory where results will be stored
results_dir =  "/home/wgq72938/Documents/Hinterer/fixed-dipole-issue/mortensen/simulation_results"

# Generate results filename with a timestamp
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
results_filename = f"Fit_results_{current_time}.csv"

# Combine the directory and filename to create the full results path
results_path = os.path.join(results_dir, results_filename)


# Ensure the results directory exists, if not, create it
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

print(f"Results will be saved to: {results_path}")

# find centroids using gaussian fitting thunderstorm
#centroids_image_coords = ground_truth_with_drawing(image_path, output_path)

# Initial guess params
initvals = array([mu, nu, background_level, photon_number, phi, theta, deltaz]) # initial PSF values
deltapix = patch_width / 2 # centre of patch around blob
initpix = (deltapix, deltapix) # centre of patch around blob
"""
# Mortensen run on each blob in each frame
x_ests, y_ests, theta_ests, phi_ests, covariance_ests = [], [], [], [], []
for i in range(tiff_stack.shape[0]):
    start_frame = time.time() # record the start time for this frame

    image = tiff_stack[i]

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #print(f"Processing frame {i + 1}/{tiff_stack.shape[0]}, image shape: {image.shape}")

    single_frame_results = list(mortensen_single_frame(image=image,
                               current_frame_number=i+1,
                               dipole_centroids=dipole_centroids,  
                               patch_width=patch_width,
                               peak_emission_wavelength=peak_emission_wavelength,
                               pixel_width=pixel_width,
                               magnification=magnification,
                               numerical_aperture=numerical_aperture,
                               #ref_ind_immersion=ref_ind_immersion,
                               ref_ind_imaging=ref_ind_imaging,
                               ref_ind_buffer=ref_ind_buffer,
                               initvals=initvals,
                               initpix=initpix,
                               deltapix=deltapix,
                               Sfloor=Sfloor,
                               inverse_gain=inverse_gain,
                               sigma_noise=sigma_noise))

    x_ests.append(single_frame_results[0])
    y_ests.append(single_frame_results[1])
    theta_ests.append(single_frame_results[2])
    phi_ests.append(single_frame_results[3])
    covariance_ests.append(single_frame_results[4])

    end_frame = time.time()
    elapsed_time_frame = end_frame - start_frame
    elapsed_time_frame = elapsed_time_frame/60
    #print(f"Time: {elapsed_time_frame:.4f} minutes on this frame")

# make sure list is flat, because it needs to be for results table
x_ests = [item for sublist in x_ests for item in sublist]
y_ests = [item for sublist in y_ests for item in sublist]
theta_ests = [item for sublist in theta_ests for item in sublist]
phi_ests = [item for sublist in phi_ests for item in sublist]
covariance_ests = [item for sublist in covariance_ests for item in sublist]

# Ensure the results directory exists
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Save all results into a single CSV file
results = pd.DataFrame({
    'x [nm]': x_ests,
    'y [nm]': y_ests,
    'theta': theta_ests,
    'phi': phi_ests,
    'covariance': covariance_ests
})
results.to_csv(results_path, index=False)
print(f"Results saved to: {results_path}")

# Read and update Thunderstorm-style results
if not os.path.exists(results_path):
    print(f"Creating a dummy results file at: {results_path}")
    dummy_df = pd.DataFrame({"x [nm]": [], "y [nm]": []})  # Replace with actual structure
    dummy_df.to_csv(results_path, index=False)

try:
    print(f"Reading results file: {results_path}")
    df = pd.read_csv(results_path)
except FileNotFoundError as e:
    print(f"Error: {e}")
    raise

if len(x_ests) != len(df) or len(y_ests) != len(df):
    raise ValueError("The length of the new x and y arrays must match the number of rows in the CSV file.")
df['x [nm]'] += x_ests
df['y [nm]'] += y_ests
df.to_csv(results_path, index=False)
"""

if __name__ == "__main__":

    # Example usage for ground truth
    image_path = "/home/wgq72938/Documents/Hinterer/fixed-dipole-issue/mortensen/image_stack_fixdip_maxproject.png"
    output_path_ground_truth = "/home/wgq72938/Documents/Hinterer/fixed-dipole-issue/mortensen/image_with_dipoles.png"
    dipole_centroids = ground_truth_with_drawing(image_path, output_path_ground_truth)
    print(f"Ground truth centroids: {dipole_centroids}")

    # Example usage for finding centroids
    default_output_path_detected = "/home/wgq72938/Documents/Hinterer/fixed-dipole-issue/mortensen/image_with_detected_centroids.png"
    detected_centroids = find_centroids(image_path, default_output_path_detected)
    print(f"Detected centroids: {detected_centroids}")

    # Compare ground truth and detected centroids
    comparison_output_path = "/home/wgq72938/Documents/Hinterer/fixed-dipole-issue/mortensen/comparison_image.png"
    compare_centroids(image_path, dipole_centroids, detected_centroids, comparison_output_path)

    # Calculate distances between ground truth and detected centroids 
    #distances = compare_centroids(dipole_centroids, detected_centroids)
    #print(f"Distances between ground truth and detected centroids: {distances}") 
    
    # Draw patches around detected centroids and store patch coordinates 
    patches_output_path = "/home/wgq72938/Documents/Hinterer/fixed-dipole-issue/mortensen/image_with_patches.png" 
    patch_coords = draw_patches(image_path, detected_centroids, patches_output_path)

    results = mortensen_single_frame(image, current_frame_number,dipole_centroids, patch_width, peak_emission_wavelength, pixel_width, magnification, numerical_aperture, ref_ind_imaging, ref_ind_buffer, initvals, initpix, deltapix, Sfloor, inverse_gain, sigma_noise)
    print(results)

    # Draw patches around detected centroids
#    patches_output_path = "/home/wgq72938/Documents/Hinterer/fixed-dipole-issue/mortensen/image_with_patches.png"
#    draw_patches(image_path, detected_centroids, patches_output_path)
