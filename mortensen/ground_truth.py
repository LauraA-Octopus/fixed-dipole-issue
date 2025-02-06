import cv2
import os
import time
import h5py
import sys
import numpy as np
from pylab import *
import pandas as pd
from argparse import Namespace
import glob
import importlib.util
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Prevents GUI issues
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
        if isinstance(result, pd.DataFrame):
            if {'x', 'y'}.issubset(result.columns):
                # Extract only (x, y) pairs
                self.centroids = list(zip(result['x'], result['y']))
                self.x_centroids = result['x'].tolist()
                self.y_centroids = result['y'].tolist()
            else:
                raise ValueError("Expected columns 'x' and 'y' in the simulation output.")
        else:
            raise TypeError("Unexpected output type from simulation3. Expected DataFrame.")

        print(f"Processed simulation3 centroids: {self.centroids}")
        print(f"All X centroids: {self.x_centroids}")
        print(f"All Y centroids: {self.y_centroids}")

        print("Custom behavior after simulation3")
        return self.centroids, self.x_centroids, self.y_centroids
    
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
                raise TimeoutError(f"file not found within {timeout} seconds.")

            time.sleep(1)  # Wait before checking again

    

    def draw_patches(self, png_file, output_path, patch_width=12):
        """ 
        Draw patches around centroids obtained from simulation3() on the image and print their coordinates. 

        Parameters:
            png_file (str): Path to the image file.
            output_path (str): Path to save the image with patches drawn.
            patch_width (int): Width of the square patches.

        Returns:
            patch_coordinates (list): List of coordinates for each patch.
        """ 
        # Ensure centroids are available
        if not hasattr(self, 'centroids') or not self.centroids:
            raise ValueError("Centroids not found. Run simulation3() first.")

        # Load the image 
        image = cv2.imread(png_file, cv2.IMREAD_GRAYSCALE) 
        if image is None: 
            raise FileNotFoundError(f"Image not found at {png_file}") 
        
        # Convert to color image for better visualization 
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) 

        patch_coordinates = [] 
        for x, y in self.centroids:  # Use self.centroids from simulation3()
            # Define the patch coordinates 
            x_start = max(0, x - patch_width // 2) 
            x_end = min(image.shape[1], x + patch_width // 2) 
            y_start = max(0, y - patch_width // 2) 
            y_end = min(image.shape[0], y + patch_width // 2) 

            # Store the patch coordinates 
            patch_coordinates.append((x_start, y_start, x_end, y_end)) 

            # Draw a rectangle around the patch 
            cv2.rectangle(image, (x_start, y_start), (x_end, y_end), color=(255, 255, 0), thickness=1) 

        # Save the image with patches 
        cv2.imwrite(output_path, image) 
        print(f"Image with patches saved to {output_path}") 
        return patch_coordinates
    
    def get_latest_dataset_directory(self, base_dir):
        """
        Returns the path of the latest dataset_* directory.
        """
        subdirs = [d for d in glob.glob(f"{base_dir}/dataset_*") if os.path.isdir(d)]
        if not subdirs:
            raise FileNotFoundError("No dataset directories found.")

        latest_subdir = max(subdirs, key=os.path.getmtime)
        return latest_subdir
        

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

        # Use self.centroids from simulation3()
        if not hasattr(self, 'centroids') or not self.centroids:
            raise ValueError("Centroids not found. Run simulation3() first.")

        # Validate centroids (skip those that are too close to the edges for a full patch)
        valid_centroids = []
        #valid_Xcentroids_nm = []
        #valid_Ycentroids_nm = []
        for x, y in centroids_image_coords:
            x_pix = x / 51
            y_pix = y / 51
            # Ensure centroid is within bounds such that the patch doesn't go out of the image.
            if patch_width // 2 <= x_pix < image.shape[1] - patch_width // 2 and \
            patch_width // 2 <= y_pix < image.shape[0] - patch_width // 2:
                valid_centroids.append((int(x_pix), int(y_pix)))  # Ensure centroids are integers
                #valid_Xcentroids_nm.append((int(x)))
                #valid_Ycentroids_nm.append((int(y)))

        if len(valid_centroids) == 0:
            print("No valid centroids after filtering.")
            return [], [], []

        # Debug: print valid centroids
        print(f"Ground truth centroids in pixels: {valid_centroids}")
        #print(f"Ground truth xcentroids: {valid_Xcentroids_nm}")
        #print(f"Ground truth ycentroids: {valid_Xcentroids_nm}")

        # Create an instance of MLEwT
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
        mortensen_results = []
        for i, (x, y) in enumerate(self.centroids, 1):
            try:
                x_pix = x / 51
                y_pix = y / 51
                # Ensure initpix is a tuple of integers
                track.initpix = (int(y_pix), int(x_pix))

                # Debug: Log inputs to Estimate
                print(f"Processing centroid {i}: initpix=({y_pix}, {x_pix})")

                # Call the Estimate method with the full image
                
                result = track.Estimate(image)  
                #mortensen_results.extend(track._observer) 
                
            except Exception as e:
                print(f"Error processing centroid {i} at ({x}, {y}): {e}")
                traceback.print_exc()

        mortensen_results.extend(track._observer)
        base_dir = "/home/wgq72938/Documents/Hinterer/fixed-dipole-issue/mortensen/simulation_results"
        latest_dataset_dir = self.get_latest_dataset_directory(base_dir)

        # Define the file path inside the latest dataset folder
        output_file = os.path.join(latest_dataset_dir, "Mortensen.csv")

        # Save the DataFrame
        dataframes = pd.DataFrame(mortensen_results, columns=["x est", "x err", "y est", "y err", "N", "b", "azim angle", "err azim", "polar ang", "err polar" ])
        dataframes.to_csv(output_file, index=False)

        print(f"Mortensen.csv saved to: {output_file}")

        return dataframes
    
    def analyse_simulation(self, png_file, **mortensen_params):
        """
        Analyse the simulation using Mortensen estimation on the extracted centroids
        """

        # Load the image
        image = cv2.imread(png_file, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found at {png_file}")
        
        # Ensure centroids are available
        if not hasattr(self, 'centroids') or not self.centroids:
            raise ValueError("Centroids not found. Run simulation3() first.")
        
        # Perform Mortensen estimation
        dataframes = self.mortensen_single_frame(
            image=image,
            current_frame_number=1,
            centroids_image_coords = self.centroids,  
            **mortensen_params
        )   

        #print(f"Results:\n X: {x_list}\n Y: {y_list}\n Theta: {theta_list}\n Phi: {phi_list}\n Covariance: {covariance_list}")
        return dataframes

if __name__ == "__main__":

    # Setup arguments and instantiate the custom application class
    args = Namespace(output="/home/wgq72938/Documents/Hinterer/fixed-dipole-issue/mortensen/simulation_results", other_arg="object")
    app_instance = CustomApplication()
    app_instance._args = args

    # Run simulation3
    app_instance.simulation3()    # centroids =

     # Ensure centroids are available
    if not hasattr(app_instance, 'centroids') or not app_instance.centroids:
        raise ValueError("Centroids not found. Ensure simulation3() sets self.centroids.")

    # Example usage for finding centroids
    png_file = app_instance.wait_for_file(file_extension="png")
    print(f"png_file type: {type(png_file)}, value: {png_file}")
    
    #default_output_path_detected = "/home/wgq72938/Documents/Hinterer/fixed-dipole-issue/mortensen/detected_centroids.png"
    #detected_centroids = app_instance.find_centroids(png_file, default_output_path_detected)
    #print(f"Detected centroids: {detected_centroids}") 
    
    
    # Draw patches around detected centroids and store patch coordinates 
    #patches_output_path = "/home/wgq72938/Documents/Hinterer/fixed-dipole-issue/mortensen/patches.png" 
    #patch_coords = app_instance.draw_patches(png_file, detected_centroids, patches_output_path)   

    # Mortensen fit
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
        "initvals": np.array([0.1, 0.1, 100, 2000, 2, 2.3]),
        "initpix": [261, 47],
        "deltapix": 6,
        "Sfloor": 100,
        "sigma_noise": 1,
        "norm_file": norm_file,
    } 
    dataframes = app_instance.analyse_simulation(png_file, **mortensen_params)
    




    
    
    #print(f"x: {x}, y: {y}, theta: {theta}, phi: {phi}, cov: {cov}")
    #fname='/home/wgq72938/Documents/Hinterer/fixed-dipole-issue/mortensen/simulation_results/dataset_250127_1359/image_stack_fixdip_simulation.hdf5'
    #h5 = h5py.File(fname)
    #h5.keys()
    #h5['ground_truth']
    #gt = np.array(h5['ground_truth'])
    #h5['ground_truth'].attrs['__columns__']
    #array(['x', 'y', 'ilk', 'id', 'channel', 'azimuth', 'polar'], dtype=object)
    #gt[:,-1]
    #print(h5['ground_truth'].attrs['__columns__'])
    #print(gt)
    
