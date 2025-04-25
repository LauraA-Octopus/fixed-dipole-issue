import numpy as np
import os
import sys
import pandas as pd
from tifffile import TiffFile
import tifffile as tiff
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
        half_size = patch_size // 2
        x_min = max(int(center_x) - half_size, 0)
        x_max = min(int(center_x) + half_size + 1, psf_image.shape[1])
        y_min = max(int(center_y) - half_size, 0)
        y_max = min(int(center_y) + half_size + 1, psf_image.shape[0])

        patch = psf_image[y_min:y_max, x_min:x_max]

        # Actual patch center in pixel coordinates
        patch_center_x = (x_min + x_max - 1) / 2
        patch_center_y = (y_min + y_max - 1) / 2

        # Convert to nm relative to full image center
        image_center_x = psf_image.shape[1] / 2
        image_center_y = psf_image.shape[0] / 2
        patch_center_x_nm = (patch_center_x - image_center_x) * self.pixel_size
        patch_center_y_nm = (patch_center_y - image_center_y) * self.pixel_size

        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        return patch, bbox, (patch_center_x_nm, patch_center_y_nm)

    def __call__(self, *args):
        posvec = np.arange(-(self.image_size[0] - 1) / 2, self.image_size[0] / 2) * self.pixel_size
        dipole_psf = np.zeros(self.image_size)

        if len(args) == 1 and isinstance(args[0], list):
            dipoles = args[0]
        elif len(args) == 5:
            dipoles = [args]
        else:
            raise ValueError("Invalid arguments to __call__: expected either (phi, theta, x, y, photons) or list of such tuples.")

        for phi, theta, x_pos_nm, y_pos_nm, n_photons in dipoles:
            for i in range(self.image_size[0]):
                for j in range(self.image_size[1]):
                    dipole_psf[j, i] += self.DD.PSF_approx(
                        posvec[i] - x_pos_nm,
                        posvec[j] - y_pos_nm,
                        phi, theta
                    )

        dipole_psf = dipole_psf / dipole_psf.sum() * sum([dp[-1] for dp in dipoles])

        return dipole_psf

    def mortensen_fit(self, dipole_patch):
        patch_size = dipole_patch.shape[0]

        # Create a patch-sezed generator for fitting
        local_generator = DipolePSFGenerator(
            image_size = (patch_size, patch_size),
            pixel_size = self.pixel_size,
            wavelength = self.wavelength,
            n_objective = self.n_objective,
            n_sample = self.n_sample,
            magnification = self.magnification,
            NA = self.NA,
            norm_file = self.norm_file,
        )

        mux_pix = np.random.uniform(self.image_size[1] / 2 - 0.5, self.image_size[1] / 2 + 0.5)
        muy_pix = np.random.uniform((self.image_size[0] - 1)/ 2 - 0.5, (self.image_size[0] - 1) / 2 + 0.5)

        #patch_size = dipole_patch.shape[0]
        #mux_nm = np.random.uniform(-self.pixel_size * patch_size / 2, self.pixel_size * patch_size / 2)
        #muy_nm = np.random.uniform(-self.pixel_size * patch_size / 2, self.pixel_size * patch_size / 2)
        mux_nm = np.random.uniform(0 - self.pixel_size / 2, 0 + self.pixel_size / 2)
        muy_nm = np.random.uniform(0 - self.pixel_size / 2, 0 + self.pixel_size / 2)

        init_new1 = np.random.uniform(-1, 1)
        init_new2 = np.random.uniform(-1, 1)
        init_new3 = np.random.uniform(0, 1)

        init_photons = np.sum(dipole_patch)
        initvals = np.array([init_new1, init_new2, init_new3, mux_nm, muy_nm, init_photons])

        track = MLEwT(initvals, local_generator)
        result = track.Estimate(dipole_patch)

        return result
    
    def get_click_positions(self, image, frame_idx):
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray', origin='lower')
        ax.set_title(f"Click once to zoom, then again to mark dipole in Frame {frame_idx}")
        coords = []
        click_stage = [0]  # use a mutable object to track click stage in closure

        def onclick(event):
            if event.xdata is not None and event.ydata is not None:
                x, y = event.xdata, event.ydata

                if click_stage[0] == 0:
                    # First click: zoom in around the clicked point
                    zoom_size = 50  # pixels half-width around click
                    x_min = max(int(x) - zoom_size, 0)
                    x_max = min(int(x) + zoom_size, image.shape[1])
                    y_min = max(int(y) - zoom_size, 0)
                    y_max = min(int(y) + zoom_size, image.shape[0])

                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(y_min, y_max)
                    click_stage[0] = 1
                    fig.canvas.draw()
                elif click_stage[0] == 1:
                    # Second click: record and draw red point
                    coords.append((x, y))
                    ax.plot(x, y, 'ro')
                    click_stage[0] = 0  # optionally reset for next dipole
                    fig.canvas.draw()

        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        fig.canvas.mpl_disconnect(cid)
        return coords

def run_mortensen_fit(start_frame, stop_frame):
    image_size = (461, 340)
    pixel_size = 51
    wavelength = 500
    n_objective = 2.17
    n_sample = 1.31
    magnification = 215
    NA = 2.17
    norm_file = "/home/wgq72938/dipolenorm.npy"
    
    patch_size = 12
    tiff_path = "/mnt/cryosil_ro/AttoDRY800/Developments_Spring_2025/2025-03-10_01_Yeast_S.Pombe+Cam1-eGFP+1-175Green70nmND/2025-03-10_01_StormData_1/2025-03-10_01_StormData_1_MMStack_Pos0.ome.tif"

    results = []
    with tiff.TiffFile(tiff_path) as tif:
        total_frames = len(tif.pages)
        print(f"Total available frames: {total_frames}")

        if start_frame < 0 or stop_frame > total_frames or start_frame >= stop_frame:
            raise ValueError(f"Invalid frame range: start={start_frame}, stop={stop_frame}, total={total_frames}")

        for frame_idx in range(start_frame, stop_frame):
            print(f"\nProcessing frame {frame_idx}")
            frame = tif.pages[frame_idx].asarray()

            psf_generator = DipolePSFGenerator(image_size, pixel_size, wavelength, n_objective, n_sample, magnification, NA, norm_file)

            click_positions = psf_generator.get_click_positions(frame, frame_idx)

            for idx, (x, y) in enumerate(click_positions):
                patch_img, bbox, patch_center_nm = psf_generator.extract_patch(frame, x, y, patch_size)

                plt.figure()
                plt.imshow(patch_img, cmap='gray', origin='lower')
                plt.title(f"Patch {idx} from Frame {frame_idx} at ({x:.1f}, {y:.1f})")
                plt.colorbar()
                plt.show()

                result = psf_generator.mortensen_fit(patch_img)
                results.append((frame_idx, idx, x, y, result))


            plt.figure()
            plt.imshow(patch_img, cmap='gray', origin='lower')
            plt.title(f"Patch from Frame {frame_idx}")
            plt.colorbar()
            plt.show()

            result = psf_generator.mortensen_fit(patch_img)
            results.append((frame_idx, result))

    return results

if __name__ == "__main__": 

    results = run_mortensen_fit(start_frame=2640, stop_frame=2652)