import cv2
import os
import time
import sys
import numpy as np
from pylab import *
import pandas as pd
from argparse import Namespace
import glob
import importlib.util
import matplotlib.pyplot as plt
from argparse import Namespace
#import diPOLE_python3
from datetime import datetime
from scipy.optimize import fmin_powell
import traceback

sys.path.append("/home/wgq72938/Documents/octo-cryoclem-smlm/third_party/MCSF2010/fixed_dipole")
from MLEwT_fixed import MLEwT

simu_path = "/home/wgq72938/Documents/octo-cryoclem-smlm/smlm/demos/demo_simulate_image_stack.py"
spec = importlib.util.spec_from_file_location("demo_simulate_image_stack", simu_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

AppClass = module.Application

class CustomApplication(AppClass):
    def __init__(self, args=None):
        super().__init__()
        if args is None:
            # Default configuration for `_args`
            self._args = type('', (), {})()  # Create a simple object
            self._args.output = "."  # Set a default path
        else:
            self._args = args

    def simulation3(self):
        print("Custom behavior before simulation3")
    
        # Call parent class simulation3
        result = super().simulation3()
        print(f"simulation3 raw result: {result}, type: {type(result)}, shape: {result.shape}")
        
        # Process result based on its structure
        if isinstance(result, np.ndarray):
            if result.ndim == 2:  # Assume 2D array where each row is [x, y]
                result = [[(x, y) for x, y in result]]  # Wrap in a list (single frame)
            elif result.ndim == 1 and len(result) == 2:  # Single point case
                result = [[(result[0], result[1])]]
            else:
                raise ValueError(f"Unexpected result structure: {result}")

        print(f"Processed simulation3 result: {result}")
        
        # Check frame count
        if len(result) < 1:
            print("Simulation returned insufficient frames")
            png_file = self.wait_for_file(file_extension="png")
            self.centroids = self.extract_centroids(file_path = png_file, 
                                                    output_dir = "/home/wgq72938/Documents/Hinterer/fixed-dipole-issue/mortensen/simulation_results") 
        else:
            self.centroids = result

        print("Custom behavior after simulation3")
        return self.centroids
    
    def wait_for_file(self, file_extension, timeout=60):
        """
        Waits for the png file to be created.
        Returns the full path to the file.
        """
        base_dir = "/home/wgq72938/Documents/Hinterer/fixed-dipole-issue/mortensen/simulation_results"
        
        start_time = time.time()

        while True:
            # Find the latest dataset_* subdirectory
            subdirs = [d for d in glob.glob(f"{base_dir}/dataset_*") if os.path.isdir(d)]
            if not subdirs:
                raise FileNotFoundError("No dataset directories found.")

            latest_subdir = max(subdirs, key=os.path.getmtime)
            file_pattern = os.path.join(latest_subdir, f"*.{file_extension}")
            files = glob.glob(file_pattern)

            if files:
                # Get the latest file with the desired extension
                latest_file = max(files, key=os.path.getmtime)
                time.sleep(2)
                return latest_file

            # Check for timeout
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                raise TimeoutError(f"TIFF file not found within {timeout} seconds.")

            time.sleep(1)  # Wait before checking again

    def find_centroids(self, png_file, output_path, threshold=15, min_area=4):
        """
        Automatically detect centroids of bright regions (dipoles) in the image.
        Parameters:
            png_file (str): Path to the fixed image containing the dipoles.
            output_path (str): Path to save the image with drawn centroids.
            threshold (int): Pixel intensity threshold for binarization.
            min_area (int): Minimum area of detected regions to consider as dipoles.
        Returns:
            centroids (list): List of (x, y) tuples representing the detected centroid locations.
        """
        # Load the fixed image
        image = cv2.imread(png_file, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found at {png_file}")

        # Threshold the image to create a binary mask
        _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

        # Find contours in the binary image
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ground_truth = []
        for contour in contours:
            if cv2.contourArea(contour) >= min_area:
                # Calculate the centroid of the contour
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    ground_truth.append((cx, cy))

                    # Draw the centroid on the image
                    cv2.circle(image, (cx, cy), radius=1, color=(0, 255, 0), thickness=-1)

        # Save the image with detected centroids
        cv2.imwrite(output_path, image)
        print(f"Image with detected centroids saved to {output_path}")

        return ground_truth

    def draw_patches(self, png_file, ground_truth, output_path, patch_width=12): 
        """ 
        Draw patches around the given centroids on the image and print their coordinates. 
        Parameters: png_file (str): Path to the image file. centroids (list): List of (x, y) tuples representing the centroids. 
        png_file (str): Path to save the image with patches drawn. 
        patch_width (int): Width of the square patches. 
        Returns: patch_coordinates (list): List of coordinates for each patch. 
        """ 
        # Load the image 
        image = cv2.imread(png_file, cv2.IMREAD_GRAYSCALE) 
        if image is None: 
            raise FileNotFoundError(f"Image not found at {png_file}") 
        
        # Convert to color image for better visualization 
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) 

        patch_coordinates = [] 
        for cx, cy in ground_truth: 
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
    def mortensen_single_frame(self, image,
                           current_frame_number,
                           centroids_image_coords,
                           patch_width,
                           peak_emission_wavelength,
                           pixel_width,
                           magnification,
                           numerical_aperture,
                           ref_ind_immersion,
                           ref_ind_buffer,
                           inverse_gain,
                           Sfloor,
                           sigma_noise,
                           initvals,
                           initpix,
                           deltapix,
                           norm_file
                           ):
        """
        Perform Mortensen estimation on a single frame given the extracted centroids.
        """
        
        # Ensure initvals is a list
        initvals = list(initvals) if isinstance(initvals, np.ndarray) else initvals
        if not isinstance(initvals, list):
            initvals = [initvals]

        # Subtract the background roughly
        image = np.clip(image - np.mean(image), 0, 255).astype(np.uint8)

        # Validate centroids (skip those that are too close to the edges for a full patch)
        valid_centroids = []
        valid_Xcentroids_nm = []
        valid_Ycentroids_nm = []
        for cx, cy in centroids_image_coords:
            # Ensure centroid is within bounds such that the patch doesn't go out of the image.
            if patch_width // 2 <= cx < image.shape[1] - patch_width // 2 and \
            patch_width // 2 <= cy < image.shape[0] - patch_width // 2:
                valid_centroids.append((int(cx), int(cy)))  # Ensure centroids are integers
                valid_Xcentroids_nm.append((int(cx*51)))
                valid_Ycentroids_nm.append((int(cy*51)))

        if len(valid_centroids) == 0:
            print("No valid centroids after filtering.")
            return [], [], [], [], []

        # Debug: print valid centroids
        print(f"Valid centroids: {valid_centroids}")
        print(f"Ground truth xcentroids: {valid_Xcentroids_nm}")
        print(f"Ground truth ycentroids: {valid_Xcentroids_nm}")

        # Create an instance of MLEwT Michael
        track = MLEwT(peak_emission_wavelength,
                                pixel_width,
                                magnification,
                                numerical_aperture,
                                ref_ind_immersion,
                                ref_ind_buffer,
                                inverse_gain,
                                Sfloor,
                                sigma_noise,
                                initvals,
                                initpix,
                                deltapix,
                                norm_file,
                                )

        # Process each centroid
        x_list, y_list, theta_list, phi_list, covariance_list = [], [], [], [], []
        for i, (cx, cy) in enumerate(valid_centroids, 1):
            try:
                # Ensure initpix is a tuple of integers
                track.initpix = (int(cy), int(cx))

                # Debug: Log inputs to Estimate
                print(f"Processing centroid {i}: initpix=({cy}, {cx})")

                # Call the Estimate method with the full image
                
                result = track.Estimate(image)     
                print(f"result length is: {len(result)}") 

                # Validate and unpack the result
                if not isinstance(result, (list, tuple)) or len(result) < 7:
                    raise ValueError(f"Unexpected Estimate result format: {result}")

                x_est, y_est, theta_est, phi_est, cov_mat = result[0], result[1], result[4], result[5], result[6:]
                x_list.append(x_est)
                y_list.append(y_est)
                theta_list.append(theta_est)
                phi_list.append(phi_est)
                covariance_list.append(cov_mat)

                # Debug: Print results for this centroid
                print(f"Centroid {i}: x={x_est}, y={y_est}, theta={theta_est}, phi={phi_est}")

            except Exception as e:
                print(f"Error processing centroid {i} at ({cx}, {cy}): {e}")
                traceback.print_exc()

        return x_list, y_list, theta_list, phi_list, covariance_list, valid_Xcentroids_nm, valid_Ycentroids_nm
    
    def analyse_simulation(self, png_file, **mortensen_params):
        """
        Analyse the simulation using Mortensen estimation on the extracted centroids
        """

        # Load the image
        image = cv2.imread(png_file, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found at {png_file}")
        
        # Perform Mortensen estimation
        x_list, y_list, theta_list, phi_list, covariance_list = self.mortensen_single_frame(
            image=image,
            current_frame_number=1,
            centroids_image_coords = detected_centroids,
            **mortensen_params
        )

        print(f"Results:\n X: {x_list}\n Y: {y_list}\n Theta: {theta_list}\n Phi: {phi_list}\n Covariance: {covariance_list}")
        return x_list, y_list, theta_list, phi_list, covariance_list

if __name__ == "__main__":

    # Setup arguments and instantiate the custom application class
    args = Namespace(output="/home/wgq72938/Documents/Hinterer/fixed-dipole-issue/mortensen/simulation_results", other_arg="object")
    app_instance = CustomApplication()
    app_instance._args = args

    # Run simulation3
    centroids = app_instance.simulation3()

    # Example usage for finding centroids
    png_file = app_instance.wait_for_file(file_extension="png")
    print(f"png_file type: {type(png_file)}, value: {png_file}")
    
    default_output_path_detected = "/home/wgq72938/Documents/Hinterer/fixed-dipole-issue/mortensen/detected_centroids.png"
    detected_centroids = app_instance.find_centroids(png_file, default_output_path_detected)
    print(f"Detected centroids: {detected_centroids}") 
    
    # Draw patches around detected centroids and store patch coordinates 
    patches_output_path = "/home/wgq72938/Documents/Hinterer/fixed-dipole-issue/mortensen/patches.png" 
    patch_coords = app_instance.draw_patches(png_file, detected_centroids, patches_output_path)   

    # Mortensen fit
    png_file = app_instance.wait_for_file(file_extension="png")
    home = os.getenv('HOME')
    norm_file = os.path.join(home, 'dipolenorm.npy')

    mortensen_params ={
        "patch_width": 12,
        "peak_emission_wavelength": 500,
        "pixel_width": 51,
        "magnification": 215,
        "numerical_aperture": 2.15,
        "ref_ind_immersion": 1.515,
        "ref_ind_buffer": 1.31,
        "inverse_gain": 0.09,
        "initvals": np.array([0.1, 0.1, 100, 20000, 2, 2.3]),
        "initpix": [261, 47],
        "deltapix": 6,
        "Sfloor": 100,
        "sigma_noise": 1,
        "norm_file": norm_file,
    } 
    x, y, theta, phi, cov = app_instance.analyse_simulation(png_file, **mortensen_params)
    #print(f"x: {x}, y: {y}, theta: {theta}, phi: {phi}, cov: {cov}")
