import numpy as np

class PSF:
    def __init__(self, params=None):
        # PSF image
        self.image = None
        self.n_pixels = 17

        # Fluorophore properties
        self.dipole = (0, 0) # inclination and azimuth converted from Dipole class
        self.position = np.array([0, 0, 0]) # position in nm
        self.n_photons = 2000
        self.shot_noise = False
        self.reduced_excitation = False
        self.stage_drift = np.array([0, 0, 0]) # relative lateral motion (look at StageDrift class)

        # Microscope setup
        self.wavelength = 500 # nm
        self.defocus = 0 # nm
        self.astigmatism = 0 # Zernike coeff
        self.objective_na = 2.17
        self.objective_focal_length = 3000 #um (converted from Length class)
        self.refractive_indices = [1.33, 2.17, 2.17] # Specimen, Intermediate, Immersion medium
        self.height_intermediate_layer = 0  # mm

        # Back focal plane
        self.phase_mask = None  # Function placeholder
        self.n_discretization_bfp = 129  # Must be an odd integer

        # Camera properties
        self.pixel_size = 108  # nm
        self.pixel_sensitivity_mask = np.ones((9, 9))  # Example mask
        self.background_noise = 0

        if params:
            self.set_parameters(params)
        
        self.create_image()

    def set_parameters(self, params):
        """Set parameters from a dictionary."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def create_image(self):
        """Generate the PSF image based on current parameters."""
        self.image = np.zeros((self.n_pixels, self.n_pixels))
        
        # Apply phase mask, aberrations, and noise (To be implemented in detail)
        
        if self.shot_noise:
            self.image = np.random.poisson(self.image)

        if self.background_noise:
            self.image += np.random.poisson(self.background_noise, self.image.shape)
    
    def read_parameters(self):
        """Return a dictionary of current PSF parameters."""
        return {
            "n_pixels": self.n_pixels,
            "dipole": self.dipole,
            "position": self.position.tolist(),
            "n_photons": self.n_photons,
            "shot_noise": self.shot_noise,
            "reduced_excitation": self.reduced_excitation,
            "wavelength": self.wavelength,
            "defocus": self.defocus,
            "astigmatism": self.astigmatism,
            "objective_na": self.objective_na,
            "objective_focal_length": self.objective_focal_length,
            "refractive_indices": self.refractive_indices,
            "height_intermediate_layer": self.height_intermediate_layer,
            "phase_mask": self.phase_mask,
            "n_discretization_bfp": self.n_discretization_bfp,
            "pixel_size": self.pixel_size,
            "pixel_sensitivity_mask": self.pixel_sensitivity_mask.tolist(),
            "background_noise": self.background_noise,
        }