def read_parameters(obj):
    """
    Reads and returns the parameters from an object.

    Parameters:
    obj: Object containing all necessary optical and camera parameters.

    Returns:
    par: Dictionary containing the extracted parameters.
    """
    par = {
        # Image properties
        "nPixels": obj.nPixels,
        
        # Fluorophore properties
        "dipole": obj.dipole,
        "position": obj.position,
        "nPhotons": obj.nPhotons,
        "shotNoise": obj.shotNoise,
        "reducedExcitation": obj.reducedExcitation,

        # Microscope setup
        "wavelength": obj.wavelength,
        "defocus": obj.defocus,
        "astigmatism": obj.astigmatism,
        "objectiveNA": obj.objectiveNA,
        "objectiveFocalLength": obj.objectiveFocalLength,
        "refractiveIndices": obj.refractiveIndices,
        "heightIntermediateLayer": obj.heightIntermediateLayer,

        # Back focal plane
        "phaseMask": obj.phaseMask,
        "nDiscretizationBFP": obj.nDiscretizationBFP,

        # Camera properties
        "pixelSize": obj.pixelSize,
        "pixelSensitivityMask": obj.pixelSensitivityMask,
        "backgroundNoise": obj.backgroundNoise,
    }

    return par