import numpy as np
import matplotlib.pyplot as plt
from MLEwT_fixed import dipdistr, MLEwT
from save_results import save_results_to_csv

class DipolePSFGenerator:
    def __init__(self, image_size, pixel_size, wavelength, n_objective, n_sample, magnification, NA, norm_file):
        self.image_size = image_size
        self.pixel_size = pixel_size
        self.wavelength = wavelength
        self.n_sample = n_sample
        self.n_objective = n_objective
        self.magnification = magnification
        self.NA = NA
        self.norm_file = norm_file
        
        
        # Load normalization file
        self.norm_data = np.load(norm_file)
        
        # Initialize dipole distribution model
        self.DD = dipdistr(wavelength, n_objective, n_sample, magnification, NA, norm_file)
    
    def __call__(self, phi, theta, x_pos, y_pos, n_photons):
        
        # Create position vector
        posvec = np.arange(-(self.image_size[0]-1)/2, self.image_size[0]/2) * self.pixel_size
        #print(f"posvec is: {posvec}")
        #print(f"The centre of posvec is: {posvec[len(posvec)//2]}")
        dipole_psf = np.zeros(self.image_size)
        
        # Generate PSF
        for i in range(self.image_size[0]):
            for j in range(self.image_size[1]):
                dipole_psf[j, i] = self.DD.PSF_approx(posvec[i] - x_pos* self.pixel_size, 
                                                      posvec[j] - y_pos * self.pixel_size,
                                                      phi, theta, 
                                                      )
        
        
        dipole_psf = dipole_psf / dipole_psf.sum() * n_photons
        
        return dipole_psf

    def mortensen_fit(self, dipole_psf):
        
        # Define parameters
        deltapix = 0
        
        # Generate the initial mux, muy in pixels around the center of the image (9, 9)
        mux_pix = np.random.uniform(self.image_size[1] / 2 - 1, self.image_size[1]/ 2 + 1)
        muy_pix = np.random.uniform((self.image_size[0] - 1)/ 2 - 1, (self.image_size[0] - 1) / 2 + 1)
        print(f"the initial vals of mux muy: {mux_pix, muy_pix}")

        # Convert to real coordinates (0, 0)
        mux_nm = np.random.uniform(0 - 1, 0 + 1)    #(mux_pix - (self.image_size[1]) / 2) * self.pixel_size
        muy_nm = np.random.uniform(0 - 1, 0 + 1)    #(muy_pix - (self.image_size[0]) / 2) * self.pixel_size
        print(f"converted real-space mux, muy: {mux_nm, muy_nm}")
        
        init_theta = np.random.uniform(0 - 0.2, 0 + 0.2)    #np.random.uniform(np.pi/2)
        init_phi = np.random.uniform (np.pi/2 - 0.2, np.pi/2 + 0.2)     #np.random.uniform(np.pi*2)

        init_photons = np.sum(dipole_psf)
        
        initvals = np.array([init_phi, init_theta, mux_nm, muy_nm, init_photons])
        print("initvals: ", initvals, "size: ", initvals.size)
        
        initpix = (self.image_size[0] // 2, self.image_size[1] // 2) # (ypix, xpix)
        print(f"initial values for the center pixel (ypixel,xpixel): {initpix}")
        
        # initvals = the initial values to start the estimate, 
        # initpix = Array of length 2 of initial values for the center pixel (ypixel,xpixel)
        # deltapix = half the array to be analysed       
        track = MLEwT(initvals, initpix, deltapix, self)
        result = track.Estimate(dipole_psf)

        return result
    
def main():

    # Define parameters
    image_size = (18, 18)  # Image dimensions
    pixel_size = 51  # nm per pixel 
    wavelength = 500  # nm
    n_objective = 2.17  # Refractive index of objective
    n_sample = 1.31  # Refractive index of sample
    magnification = 215  # Objective magnification
    NA = 2.17  # Numerical Aperture
    norm_file = "/home/wgq72938/dipolenorm.npy"
    n_photons = 20000
    
    # Create PSF generator instance
    psf_generator = DipolePSFGenerator(image_size, pixel_size,
                                       wavelength, n_objective, n_sample,
                                       magnification, NA, norm_file)

    # Define theta and phi for simulated dipole
    theta = 0  #np.pi/2  
    phi   = np.pi/2  

    # Define simulated dipole position
    x_pos = 0  #np.random.uniform(-image_size[1]//2, image_size[1]//2)
    y_pos = 0  #np.random.uniform(-image_size[0]//2, image_size[0]//2) 

    # Generate the dipole PSF
    dipole_psf = psf_generator(phi, theta, x_pos, y_pos, n_photons)

    # Create a figure with a single subplot
    plt.figure(figsize=(5, 5))  
    plt.imshow(dipole_psf, cmap='gray', interpolation='nearest')
    plt.title(f"Theta = {theta:.2f}, Phi = {phi:.2f}\nX = {x_pos:.2f}, Y = {y_pos:.2f}")
    plt.axis('off')
    #plt.show()

    # Run Mortensen fit for an example theta=0 and phi=0
    results = psf_generator.mortensen_fit(dipole_psf)
    print(f"Results from the Mortensen fit are: {results}")
    print(f"Ground truth are: {phi}, {theta}, {x_pos}, {y_pos}, {n_photons}")

    # Save results to csv file
    save_results_to_csv(results, (phi, theta, x_pos, y_pos, n_photons))

    

if __name__ == "__main__":
    main()
