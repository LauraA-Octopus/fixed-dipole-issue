import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# File path
folder = os.path.expanduser('~/Documents/Hinterer/fixed-dipole-issue/mortensen_laura')
plot_folder = os.path.join(folder, 'plots/2000Photons')
os.makedirs(plot_folder, exist_ok=True)
file_path = os.path.join(folder, "results_phi_264.csv")

# Extract angle and type (phi or theta) from file name
filename_parts = os.path.basename(file_path).split('_')
angle_type = filename_parts[1]  # Either 'phi' or 'theta'
angle_value = filename_parts[2].split('.')[0]  # Extracts the angle value
title_angle = f"{angle_type}{angle_value}"  # e.g., "phi0" or "theta45"

# Read the CSV file
results = pd.read_csv(file_path)

# Calculate x_err and y_err
x_err = results['x_est'] - results['x_true']
y_err = results['y_est'] - results['y_true']

# Extract angles (phi and theta)
phi = results['phi_est']
theta = results['theta_est']

# Calculate parallel and perpendicular errors
parallel_error = x_err * np.cos(phi) - y_err * np.sin(phi)
perpendicular_error = x_err * np.sin(theta) + y_err * np.cos(theta)

# Add new columns to the original DataFrame
results['parallel_error'] = parallel_error
results['perpendicular_error'] = perpendicular_error

# Calculate theta_err and phi_err
theta_err = results['theta_true'] - results['theta_est']
phi_err = results['phi_true'] - results['phi_est']

# Save the updated CSV file
results.to_csv(file_path, index=False)
print(f"Updated results saved to: {file_path}")

# Function to save plots dynamically
def save_plot():
    xlabel = plt.gca().get_xlabel()
    ylabel = plt.gca().get_ylabel()
    filename = f"{title_angle}_{ylabel.replace(' ', '_')}_vs_{xlabel.replace(' ', '_')}.png"
    plt.savefig(os.path.join(plot_folder, filename))
    plt.show()

# Scatter plot: parallel_error vs theta
plt.figure(figsize=(8, 5))
plt.scatter(results['theta_est'], results['parallel_error'], color='blue', alpha=0.6)
plt.xlabel('theta')
plt.ylabel('Parallel Error')
plt.title(f'{angle_type} = {angle_value} theta = [0, 1, 2, ..., 90]')
plt.grid(True)
save_plot()

# Scatter plot: perpendicular_error vs theta
plt.figure(figsize=(8, 5))
plt.scatter(results['theta_est'], results['perpendicular_error'], color='red', alpha=0.6)
plt.xlabel('theta')
plt.ylabel('Perpendicular Error')
plt.title(f'{angle_type} = {angle_value} theta = [0, 1, 2, ..., 90]')
plt.grid(True)
save_plot()

# Scatter plot: perpendicular_error vs parallel_error
plt.figure(figsize=(8, 5))
plt.scatter(results['parallel_error'], results['perpendicular_error'], color='purple', alpha=0.6)
plt.xlabel('Parallel Error')
plt.ylabel('Perpendicular Error')
plt.title(f'{angle_type} = {angle_value} theta = [0, 1, 2, ..., 90]')
plt.grid(True)
save_plot()

# Histogram: Frequency vs parallel_error
plt.figure(figsize=(8, 5))
plt.hist(results['parallel_error'], bins=30, color='blue', alpha=0.7, edgecolor='black')
plt.xlabel('Parallel Error')
plt.ylabel('Frequency')
plt.title(f'{angle_type} = {angle_value} theta = [0, 1, 2, ..., 90]')
plt.grid(True)
save_plot()

# Histogram: Frequency vs perpendicular_error
plt.figure(figsize=(8, 5))
plt.hist(results['perpendicular_error'], bins=30, color='red', alpha=0.7, edgecolor='black')
plt.xlabel('Perpendicular Error')
plt.ylabel('Frequency')
plt.title(f'{angle_type} = {angle_value} theta = [0, 1, 2, ..., 90]')
plt.grid(True)
save_plot()

# Plot histogram for theta_err
#plt.figure(figsize=(12, 6))

#plt.subplot(1, 2, 1)
#plt.hist(theta_err, bins=30, color='blue', alpha=0.7)
#plt.title('Histogram of theta_err')
#plt.xlabel('theta_err')
#plt.ylabel('Frequency')
#plt.grid(True)
#save_plot()

# Plot histogram for phi_err
#plt.subplot(1, 2, 2)
#plt.hist(phi_err, bins=30, color='red', alpha=0.7)
#plt.title('Histogram of phi_err')
#plt.xlabel('phi_err')
#plt.ylabel('Frequency')
#plt.grid(True)
#save_plot()
