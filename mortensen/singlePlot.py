import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# File paths
folder = '~/Documents/Hinterer/fixed-dipole-issue/mortensen/simulation_results/dataset_250128_1706'
ground_truth_file = os.path.join(folder, "image_stack_fixdip_ground_truth.csv")
mortensen_file = os.path.join(folder, "Mortensen.csv")

# Read the CSV files
ground_truth = pd.read_csv(ground_truth_file)
mortensen = pd.read_csv(mortensen_file)

# Adjust lengths if they differ
min_length = min(len(ground_truth), len(mortensen))
ground_truth = ground_truth.iloc[:min_length]  # Trim to min_length
mortensen = mortensen.iloc[:min_length]        # Trim to min_length

# Calculate x_err and y_err
x_err = ground_truth['x'] - mortensen['y est']
y_err = ground_truth['y'] - mortensen['x est']

# Extract angles (phi and theta) from Mortensen dataset
phi = ground_truth['azimuth']
theta = ground_truth['polar']

# Calculate parallel and perpendicular errors
parallel_error = x_err * np.cos(phi) - y_err * np.sin(phi)
perpendicular_error = x_err * np.sin(theta) + y_err * np.cos(theta)

# Add errors to a new DataFrame for convenience
errors_df = pd.DataFrame({
    'Dipole': range(1, len(x_err) + 1),  # Assuming dipoles indexed from 1
    'x_err': x_err,
    'y_err': y_err,
    'parallel_error': parallel_error,
    'perpendicular_error': perpendicular_error
})

# Automatically increment file name
def get_next_file_name(folder, base_name, extension):
    i = 1
    while True:
        file_name = f"{base_name}_{i}.{extension}"
        if not os.path.exists(os.path.join(folder, file_name)):
            return os.path.join(folder, file_name)
        i += 1

# Determine next available file name
errors_output_file = get_next_file_name(folder, "parallel_perpendicular_errors", "csv")

# Save errors to a CSV file
errors_df.to_csv(errors_output_file, index=False)
print(f"Parallel and perpendicular errors saved to: {errors_output_file}")

# Scatterplot: x_err as a function of dipoles
plt.figure(figsize=(10, 5))
plt.scatter(errors_df['Dipole'], errors_df['x_err'], color='blue', label='x_err')
plt.xlabel('Dipole')
plt.ylabel('x_err')
plt.title('x_err as a Function of Dipoles')
plt.legend()
plt.grid(True)
#plt.savefig(os.path.join(folder, "x_err_scatterplot.png"))
plt.show()

# Scatterplot: y_err as a function of dipoles
plt.figure(figsize=(10, 5))
plt.scatter(errors_df['Dipole'], errors_df['y_err'], color='red', label='y_err')
plt.xlabel('Dipole')
plt.ylabel('y_err')
plt.title('y_err as a Function of Dipoles')
plt.legend()
plt.grid(True)
#plt.savefig(os.path.join(folder, "y_err_scatterplot.png"))
plt.show()

# Scatterplot: x_err as a function of dipoles
plt.figure(figsize=(10, 5))
plt.scatter(errors_df['Dipole'], errors_df['parallel_error'], color='blue', label='||_err')
plt.xlabel('Dipole')
plt.ylabel('||_err')
plt.title('||_err as a Function of Dipoles')
plt.legend()
plt.grid(True)
#plt.savefig(os.path.join(folder, "x_err_scatterplot.png"))
plt.show()

# Scatterplot: x_err as a function of dipoles
plt.figure(figsize=(10, 5))
plt.scatter(errors_df['Dipole'], errors_df['perpendicular_error'], color='blue', label='|-_err')
plt.xlabel('Dipole')
plt.ylabel('|-_err')
plt.title('|-_err as a Function of Dipoles')
plt.legend()
plt.grid(True)
#plt.savefig(os.path.join(folder, "x_err_scatterplot.png"))
plt.show()
