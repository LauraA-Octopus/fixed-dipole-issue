import numpy as np
import matplotlib.pyplot as plt
from MLEwT_fixed import dipdistr

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
        print("Loaded norm file shape:", self.norm_data.shape)
        
        # Initialize dipole distribution model
        self.DD = dipdistr(wavelength, n_objective, n_sample, magnification, NA, norm_file)
    
    def __call__(self, phi, theta, x_pos, y_pos, n_photons):
        # Create position vector
        posvec = np.arange(-(self.image_size[0]-1)/2, self.image_size[0]/2) * self.pixel_size
        dipole_psf = np.zeros(self.image_size)
        
        # Generate PSF
        for i in range(self.image_size[0]):
            for j in range(self.image_size[1]):
                dipole_psf[j, i] = self.DD.PSF_approx(posvec[i] - x_pos * self.pixel_size, 
                                                       posvec[j] - y_pos * self.pixel_size, 
                                                       phi, theta)
        
        
        dipole_psf = dipole_psf / dipole_psf.sum() * n_photons
        
        return dipole_psf
        
def mortensen_fit():
    
    psf_generator = DipolePSFGenerator(image_size, pixel_size, wavelength, n_objective, n_sample, magnification, NA, norm_file)
    datamatrix= psf_generator(theta, phi, image_size[1]//2, image_size[0]//2)
    track=MLEwT(psf_generator)
    track.Estimate(datamatrix)


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

    
    # Create PSF generator instance
    psf_generator = DipolePSFGenerator(image_size, pixel_size, 
                                       wavelength, n_objective, n_sample,
                                       magnification, NA, norm_file)

# Define theta values to span between 0 and 1.57 radians
    theta_values = np.linspace(0, 1.57, 9)  # 9 different theta values
    phi = 0  # Keep phi constant
    #x_pos, y_pos = image_size[1] // 2, image_size[0] // 2  # Centered position
    
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))  # 3x3 grid of images
    
    for ax, theta in zip(axes.flat, theta_values):
        dipole_psf = psf_generator(phi, theta, x_pos=0, y_pos=0, n_photons=500)
        dipole_psf_noisy = np.random.poisson(dipole_psf)
        ax.imshow(dipole_psf, cmap='gray', interpolation='nearest')
        ax.set_title(f"Theta = {theta:.2f}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    

if __name__ == "__main__":
    main()
