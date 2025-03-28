import numpy as np
import tifffile as tiff
import os

# TODO: Need to define classes like Length, Dipole and PSF

# Set parameters
inclinations = np.arange(0, np.pi/2 + 22.5 * (np.pi / 180), 22.5 * (np.pi / 180)) # theta
azimuths = np.arange(0, 2 * np.pi - 1 * (np.pi / 180), 4 * (np.pi / 180))  #phi

number_of_spots = 1
scalefactor = 1
padding = 0.15
inner_bound = padding
outer_bound = 1 - 2 * padding
pixel_size_nm = 52 / scalefactor
image_size_nm = np.sqrt(number_of_spots) * 1000
image_size_px = int(image_size_nm / pixel_size_nm) | 1 # ensure odd

wavelength = 500
objectiveFocalLength = 770

par = {
    "nPixels": image_size_px,
    "wavelength": wavelength,
    "objectiveNA": 2.17,
    "objectiveFocalLength": objectiveFocalLength,
    "refractiveIndices": [1.31, 2.17, 2.17],
    "nDiscretizationBFP": 129,
    "pixelSize": pixel_size_nm,
    "pixelSensistivityMask": np.ones((9, 9)),
    "nPhotons": 2000,
}
backgroundNoise = 0

output_dir = "/home/wgq72938/Documents/Hinterer/fixed-dipole-issue/mortensen_laura/results/hinterer_generator"
os.makedirs(output_dir, exist_ok=True)

runs = range(1, 6)
for run in runs:
    for inclination in inclinations:
        for azimuth in azimuths:
            inclination_deg = np.degrees(inclination)
            azimuth_deg = np.degrees(azimuth)
            print(f"Running inc={inclination_deg: .2f} az={azimuth_deg: .2f}")

            output_path = os.path.join(output_dir, f"sim_inc{int(inclination_deg): 03d}_az{int(azimuth_deg):03d}_run{run}.tif")
            data_output_path = os.path.join(output_dir, f"params_inc{int(inclination_deg):03d}_az{int(azimuth_deg):03d}_run{run}.txt")

            positionX_nm_array, positionY_nm_array = [], []
            angleInclination_array, angleAzimuth_array = [], []

            for i in range(number_of_spots):
                positionX_nm = -pixel_size_nm/2 + np.random.rand() * pixel_size_nm
                positionY_nm = -pixel_size_nm/2 + np.random.rand() * pixel_size_nm

                angleInclination = inclination
                angleAzimuth = azimuth

                par["position"] = [positionX_nm, positionY_nm, 0]
                par["dipole"] = Dipole(angleInclination, angleAzimuth)

                if i == number_of_spots - 1:
                    par["backgroundNoise"] = backgroundNoise
                    par["shotNoise"] = 1
                else:
                    par["backgroundNoise"] = 0

                psf = np.random.rand(image_size_px, image_size_px) # Placeholder for actual psf calculation
                if i == 0:
                    psf_total_image = psf
                else:
                    psf_total_image += psf

                positionX_nm_array.append(positionX_nm)
                positionY_nm_array.append(positionY_nm)
                angleInclination_array.append(angleInclination)
                angleAzimuth_array.append(angleAzimuth)

            # Save the image
            psf_total_image = (psf_total_image * (2**32 - 1)).astype(np.uint32)
            tiff.imwrite(output_path, psf_total_image, photometric='minisblank', compression='lzw')

            # Normalize for display
            display_image = (psf_total_image - psf_total_image.min()) / (psf_total_image.max() - psf_total_image.min())
            tiff.imwrite(output_path.replace('.tif', '_display.tif'), (display_image * 255).astype(np.uint8))
            print(f"Simulation output to \n {output_path}")

            # Save parameters
            with open(data_output_path, 'w') as file:
                file.write(f"# Ground truth for sim_inc{int(inclination_deg):03d}_az{int(azimuth_deg):03d}_run{run}.tif\n")
                file.write(f"number_of_spots = {number_of_spots}\n")
                file.write(f"pixel_size_no = {pixel_size_nm: 3f}\n")
                file.write(f"image_size_nm = {image_size_nm: 3f}\n")
                file.write(f"image_size_px = {image_size_px}\n")
                file.write(f"wavelength = {wavelength}\n")
                file.write(f"par.objectiveNA = {par['objectiveNA']: .2f}\n")
                file.write(f"objectiveFocalLength = {objectiveFocalLength}\n")
                file.write(f"par.refractiveIndices = {par['refractiveIndices']}\n") 
                file.write(f"nDiscretizationBFP = {par['nDiscretizationBFP']}\n") 
                file.write(f"backgroundNoise = {par['backgroundNoise']}\n") 
                file.write(f"par.nPhotons = {par['nPhotons']}\n") 
                file.write(f"positionX_nm_array = {positionX_nm_array}\n") 
                file.write(f"positionY_nm_array = {positionY_nm_array}\n") 
                file.write(f"angleInclination_array = {angleInclination_array}\n") 
                file.write(f"angleAzimuth_array = {angleAzimuth_array}\n") 