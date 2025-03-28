import numpy as np 
import sys
import datetime
import matplotlib.pyplot as plt
from MLEwT_fixed import dipdistr, MLEwT
from oct2py import Oct2Py

class DipolePSFSimulator:
    def __init__(self, matlab_script_path):
        self.oc = Oct2Py()
        self.oc.addpath(matlab_script_path)

    def simulate(self, output_folder, inclination_deg, azimuth_deg, run):
        """
        Calls the MATLAB function to generate a dipole PSF.

        :param output_folder: Path to save the output images
        :param inclination_deg: Inclination angle in degrees
        :param azimuth_deg: Azimuth angle in degrees
        :param run: Run number
        :return: Path of the generated image
        """
        output_path = self.oc.generator_pyFunction(output_folder, inclination_deg, azimuth_deg, run)
        return output_path

# Example usage
if __name__ == "__main__":
    matlab_folder = '/home/wgq72938/Documents/Hinterer/fixed-dipole-issue/fixed-dipole-issue/hinterer'
    output_folder = '/home/wgq72938/Documents/Hinterer/fixed-dipole-issue/mortensen_laura/results/hinterer_generator'

    simulator = DipolePSFSimulator(matlab_folder)
    output_file = simulator.simulate(output_folder, 45, 90, 1)
    print("Generated PSF image:", output_file)

    
'''

    def mortensen_fit(self, dipole_psf, init_theta, init_phi):
        
        # Define parameters
        #deltapix = 9
        
        # Generate the initial mux, muy in pixels around the center of the image (9, 9)
        mux_pix = np.random.uniform(self.image_size[1] / 2 - 0.5, self.image_size[1]/ 2 + 0.5)
        muy_pix = np.random.uniform((self.image_size[0] - 1)/ 2 - 0.5, (self.image_size[0] - 1) / 2 + 0.5)
        #print(f"the initial vals of mux muy: {mux_pix, muy_pix}")

        # Convert to coordinates in nm
        mux_nm = np.random.uniform(0 - self.pixel_size/2, 0 + self.pixel_size/2)    
        muy_nm = np.random.uniform(0 - self.pixel_size/2, 0 + self.pixel_size/2)    
        
        init_theta = np.random.uniform(0, np.pi/2)
        init_phi = np.random.uniform(0, 2 * np.pi)

        init_photons = np.sum(dipole_psf)
        
        initvals = np.array([init_phi, init_theta, mux_nm, muy_nm, init_photons])
        
        #initpix = (self.image_size[0] // 2, self.image_size[1] // 2) # (ypix, xpix)
        #print(f"initial values for the center pixel (ypixel,xpixel): {initpix}")
        
        # initvals = the initial values to start the estimate       
        track = MLEwT(initvals, self)   #initpix, deltapix, self)
        result = track.Estimate(dipole_psf)

        return result

def run_mortensen_fit(phi, theta):
    """
    Runs the Mortensen fit for given phi and theta, returning the results and ground truth.
    """
    # Define parameters
    image_size = (18, 18)  
    pixel_size = 51  
    wavelength = 500  
    n_objective = 2.17  
    n_sample = 1.31  
    magnification = 215  
    NA = 2.17  
    norm_file = "/home/wgq72938/dipolenorm.npy"
    n_photons = 2000

    # Define dipole ground_truth position in nm
    x_pos_nm = np.random.uniform(0 - pixel_size/2, 0 + pixel_size/2)  
    y_pos_nm = np.random.uniform(0 - pixel_size/2, 0 + pixel_size/2) 
    
    # Create PSF generator instance
    psf_generator = DipolePSFGenerator(image_size, pixel_size, wavelength, n_objective, n_sample, magnification, NA, norm_file)    

    # Generate dipole PSF
    dipole_psf = psf_generator(phi, theta, x_pos_nm, y_pos_nm, n_photons)
    dipole_psf_noisy = np.random.poisson(dipole_psf)

    # Run Mortensen fit
    results = psf_generator.mortensen_fit(dipole_psf_noisy, theta, phi)
    
    # Return results instead of printing
    return results, [phi, theta, x_pos_nm, y_pos_nm, n_photons]

# Prevent script from running when imported
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python test.py <phi> <theta>")
        sys.exit(1)

    phi = float(sys.argv[1])
    theta = float(sys.argv[2])

    results, ground_truth = run_mortensen_fit(phi, theta)
    
    print(f"Results from the Mortensen fit are:  {', '.join(map(str, results))}")    
    print(f"Ground truth are: {', '.join(map(str, ground_truth))}")

'''