import os
import csv
from real_data import run_mortensen_fit

# ==== USER SETTINGS ====
start_frame = 2640
stop_frame = 2652
output_folder = "~/Documents/Hinterer/fixed-dipole-issue/mortensen_laura/multiDipoles/plots/test_patches"
output_name = "fitting_results.csv"
# ========================

def save_results_to_csv(results, output_folder, output_name):
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_name)

    with open(output_path, mode="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame', 'Dipole', 'Phi_gt', 'Theta_gt', 'X_gt', 'Y_gt', 'Phi_est', 'Theta_est', 'X_est', 'Y_est'])

        for frame_idx, dipole_idx, x, y, result in results:
            phi_gt, theta_gt = result[0], result[1]
            x_gt, y_gt = x, y

            phi_est, theta_est, x_est, y_est = result

            writer.writerow([frame_idx, dipole_idx, phi_gt, theta_gt, x_gt, y_gt, phi_est, theta_est, x_est, y_est])

    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    print(f"Running Mortensen fitting on frames {start_frame} to {stop_frame -1}...")
    results = run_mortensen_fit(start_frame=start_frame, stop_frame=stop_frame)
    save_results_to_csv(results, output_folder, output_name)

