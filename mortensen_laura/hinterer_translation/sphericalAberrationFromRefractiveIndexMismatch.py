import numpy as np 

def spherical_aberration_from_mismatch(obj, remove_defocus=False):
    """
    Computes spherical aberration due to refractive index mismatch.

    Parameters:
    obj: Object containingoptical parameters.
    remove_defocus: Boolean flag to remove defocus analytically.

    Returns:
    Defocus: Spherical aberration phase (for 1m inside material)
    SA: Spherical aberration phase
    a: Coefficient of defocus contained in SA
    """

    N = obj.nDiscretizationBFP
    NA = obj.objectiveNA
    n2 = obj.refractiveIndices[2]
    k = 2 * np.pi / obj.wavelength.inMeter # this requires further python implementations

    par = {
        'nGrid': N,
        'spacing': 2 / (N - 1),
        'mode': 'FFT'
    }

    mask = Mask(par) # this class needs to be transformed from Matlab
    normR, _ = get_polar_coordinates(mask)

    # Mean value of spherical defocus
    MW = lambda RI: (2 / 3) * k * (-(-1 + (RI**2 / NA**2))**(3/2) + (RI**2 / NA**2)**(3/2)) * NA

    # Defocus function
    Def = lambda RI: np.real(k * NA * np.sqrt(RI**2 / NA**2 - normR**2) - MW(RI)) * mask.values
    Defocus = Def(n2)

    if remove_defocus:
        SA = None
        a = None
    else:
        n1 = obj.refractiveIndices[2]  # Using n2 instead of n1 (possible typo in MATLAB code?)
        SA = -(Def(n2) - Def(n1))

        a_spher = (-1/72 * k**2 * np.pi * 
                   (72 * n2**2 - 36 * NA**2 +
                    (32 * (-(-1 + n2**2 / NA**2)**(3/2) + n2**3 / NA**3) *
                     (n1**3 - n1**2 * np.sqrt((n1 - NA) * (n1 + NA)) + NA**2 * np.sqrt((n1 - NA) * (n1 + NA)))) / NA -
                    (32 * (n2**3 - n2**2 * np.sqrt((n2 - NA) * (n2 + NA)) + NA**2 * np.sqrt((n2 - NA) * (n2 + NA)))**2) / NA**4 +
                    (9 / NA**2) * (2 * (-n1**3 * n2 - n1 * n2**3 + 
                                        n1**2 * np.sqrt((n1 - NA) * (n2 - NA) * (n1 + NA) * (n2 + NA)) + 
                                        np.sqrt((n1 - NA) * (n2 - NA) * (n1 + NA) * (n2 + NA)) * (n2**2 - 2 * NA**2)) -
                                   (n1**2 - n2**2)**2 * 
                                   (np.log((n1 - n2)**2) - np.log(n1**2 + n2**2 - 2 * (NA**2 + np.sqrt((n1 - NA) * (n2 - NA) * (n1 + NA) * (n2 + NA))))))))

        def_norm = -((k**2 * (16 * n2**6 - 24 * n2**4 * NA**2 + 6 * n2**2 * NA**4 + NA**6 - 
                              16 * n2**5 * np.sqrt(n2**2 - NA**2) + 
                              16 * n2**3 * NA**2 * np.sqrt(n2**2 - NA**2)) * np.pi) / (18 * NA**4))
        
        a = a_spher / def_norm

        if remove_defocus:
            SA -= a_spher * Def(n2) / def_norm

    return Defocus, SA, a

