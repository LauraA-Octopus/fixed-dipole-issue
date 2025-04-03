import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from MLEwT_fixed import MLEwT, dipdistr

class DipolePSFGenerator:
    def __init__(self, image_size, pixel_size, wavelength, n_objective, n_sample, magnification, NA, norm_file):
        self.image_size = image_size
        self.pixel_size = pixel_size
        self.wavelength = wavelength
        self.n_objective = n_objective
        self.n_sample = n_sample
        self.magnification = magnification
        self.NA = NA
        self.norm_file = norm_file

        # Load norm file
        self.norm_data = np.load(norm_file)

        # Initialize dipole distribution model
        self.DD = dipdistr(wavelength, n_objective, n_sample, magnification, NA, norm_file)

    def extract_patch(self, psf_image, center_x, center_y, patch_size):
        """Extracts a square patch from the PSF image centered around (center_x, center_y)."""
        half_size = patch_size // 2
        
        # Ensure the center is within the patch bounds (i.e., the dipole is within 1 pixel of the patch center)
        x_min = int(np.round(center_x - half_size))
        x_max = int(np.round(center_x + half_size))
        y_min = int(np.round(center_y - half_size))
        y_max = int(np.round(center_y + half_size))

        # Ensure the patch does not go out of bounds of the image
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(x_max, psf_image.shape[1])
        y_max = min(y_max, psf_image.shape[0])

        # Extract the patch and return it with its rectangle coordinates
        extracted_patch = psf_image[y_min:y_max, x_min:x_max]
        
        return extracted_patch, (x_min, y_min, x_max - x_min, y_max - y_min)
        
    def __call__(self, phi, theta, x_pos_nm, y_pos_nm, n_photons):
        
        # Create position vector
        posvec = np.arange(-(self.image_size[0] - 1) / 2, self.image_size[0] / 2) * self.pixel_size
        dipole_psf = np.zeros(self.image_size)

        # Generate PSF
        for i in range(self.image_size[0]):
            for j in range(self.image_size[1]):
                dipole_psf[j, i] = self.DD.PSF_approx(posvec[i] - x_pos_nm, 
                                                      posvec[j] - y_pos_nm,
                                                      phi, theta)
                
        dipole_psf = dipole_psf / dipole_psf.sum() * n_photons

        # Ensure the patch is centered within 1 pixel of the dipole
        center_x_pix = self.image_size[1] // 2 + (x_pos_nm / self.pixel_size)
        center_y_pix = self.image_size[0] // 2 + (y_pos_nm / self.pixel_size)
        patch_size = 18  # Adjust as needed

        dipole_patch, patch_rect = self.extract_patch(dipole_psf, center_x_pix, center_y_pix, patch_size)

        #print(f"Extracted patch shape: {dipole_patch.shape}")
        
        # Visualize
        #fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        #ax1, ax2 = axes
        
        # Full PSF with patch overlay
        #ax1.imshow(dipole_psf, cmap='hot', origin='lower')
        #rect = patches.Rectangle((patch_rect[0], patch_rect[1]), patch_rect[2], patch_rect[3], linewidth=1.5, edgecolor='cyan', facecolor='none')
        #ax1.add_patch(rect)
        #ax1.set_title("Full PSF with patch overlay")

        # Extracted patch
        #ax2.imshow(dipole_patch, cmap='hot',origin='lower')
        #ax2.set_title("extracted patch")

        #plt.show()

        return dipole_patch  # Instead of full dipole_psf
    
    def mortensen_fit(self, dipole_patch):
        
        # Generate initial positions in pixels around the center of the image
        mux_pix = np.random.uniform(self.image_size[1] / 2 - 0.5, self.image_size[1] / 2 + 0.5)
        muy_pix = np.random.uniform((self.image_size[0] - 1)/ 2 - 0.5, (self.image_size[0] - 1) / 2 + 0.5)

        # Convert to coordinates in nm
        mux_nm = np.random.uniform(0 - self.pixel_size/2, 0 + self.pixel_size/2)
        muy_nm = np.random.uniform(0 - self.pixel_size/2, 0 + self.pixel_size/2)
        
        init_new1 = np. random.uniform(-1, 1)
        init_new2 = np.random.uniform(-1, 1)
        init_new3 = np.random.uniform(0, 1)

        init_photons = np.sum(dipole_patch)
        
        initvals = np.array([init_new1, init_new2, init_new3, mux_nm, muy_nm, init_photons])
        print(f"Initvals = {initvals}")
        
        track = MLEwT(initvals, self)
        result = track.Estimate(dipole_patch)

        return result

def run_mortensen_fit(phi, theta):
    
    # Define parameters
    image_size = (25, 25)
    pixel_size = 51
    wavelength = 500
    n_objective = 2.17
    n_sample = 1.31
    magnification = 215
    NA = 2.17
    norm_file = "/home/wgq72938/dipolenorm.npy"
    n_photons = 2000

    # Define dipole ground truth position in nm
    x_pos_nm = np.random.uniform(0 - pixel_size/2, 0 + pixel_size/2)
    y_pos_nm = np.random.uniform(0 - pixel_size/2, 0 + pixel_size/2)

    psf_generator = DipolePSFGenerator(image_size, pixel_size, wavelength, n_objective, n_sample, magnification, NA, norm_file)
    dipole_psf = psf_generator(phi, theta, x_pos_nm, y_pos_nm, n_photons)
    dipole_psf_noisy = np.random.poisson(dipole_psf)
    
    # Run Mortensen fit
    results = psf_generator.mortensen_fit(dipole_psf_noisy)

    return results, [phi, theta, x_pos_nm, y_pos_nm, n_photons]

if __name__ == "__main__": 
    if len(sys.argv) != 3:
        print("Usage: python test_onRealData.py <phi> <theta>")
        sys.exit(1)

    phi = float(sys.argv[1])
    theta = float(sys.argv[2])

    results, ground_truth = run_mortensen_fit(phi, theta)
    
    print(f"Results from the Mortensen fit are: {' , '.join(map(str, results))}")
    print(f"Ground truth are: {' , '.join(map(str, ground_truth))}")