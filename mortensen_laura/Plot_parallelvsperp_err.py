import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# File path
folder = os.path.expanduser('~/Documents/Hinterer/fixed-dipole-issue/mortensen_laura')
plot_folder = os.path.join(folder, 'plots/')
os.makedirs(plot_folder, exist_ok=True)
file_path = os.path.join(folder, "results_theta_67.5.csv")

# Extract angle and type (phi or theta) from file name
filename_parts = os.path.basename(file_path).split('_')
angle_type = filename_parts[1]  
angle_value = filename_parts[2].split('.')[0]  
title_angle = f"{angle_type}{angle_value}"  

# Read the CSV file
results = pd.read_csv(file_path)

# Calculate x_err and y_err
x_err = results['x_est'] - results['x_true']
y_err = results['y_est'] - results['y_true']


# Extract angles (phi and theta)
phi = results['phi_est']
theta = results['theta_est']

phi_true = results['phi_true']
theta_true = results['theta_true']

# Calculate parallel and perpendicular errors
parallel_error = x_err * np.cos(phi_true) - y_err * np.sin(phi_true)
perpendicular_error = x_err * np.sin(phi_true) + y_err * np.cos(phi_true)

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

# Scatter plot: x coords nm
plt.figure(figsize=(8, 5))
plt.scatter(results['x_true'], results['x_est'], color='blue', alpha=0.6)
plt.xlabel('x_true nm')
plt.ylabel('x_est nm')
plt.title(f'{angle_type} = {angle_value} theta = [0, 2, 4, ..., 90]')
plt.grid(True)
save_plot()

# Scatter plot: y coords nm
plt.figure(figsize=(8, 5))
plt.scatter(results['y_true'], results['y_est'], color='blue', alpha=0.6)
plt.xlabel('y_true nm')
plt.ylabel('y_est nm')
plt.title(f'{angle_type} = {angle_value} phi = [0, 4, 8, ..., 360]')
plt.grid(True)
save_plot()

# Scatter plot: coords pix
#plt.figure(figsize=(8, 5))
#plt.scatter(results['x_true']/51, results['x_est']/51, color='blue', alpha=0.6)
#plt.xlabel('x_true [pix]')
#plt.ylabel('x_est [pix]')
#plt.title(f'{angle_type} = {angle_value} theta = [0, 2, 4, ..., 90]')
#plt.grid(True)
#save_plot()

# Scatter plot: angles rad
plt.figure(figsize=(8, 5))
plt.scatter(results['phi_true'], results['phi_est'], color='blue', alpha=0.6)
plt.xlabel('phi_true rad')
plt.ylabel('phi_est rad')
plt.title(f'{angle_type} = {angle_value} phi = [0, 4, 8, ..., 360]')
plt.grid(True)
save_plot()

# Scatter plot: angles rad
plt.figure(figsize=(8, 5))
plt.scatter(results['theta_true'], results['theta_est'], color='blue', alpha=0.6)
plt.xlabel('theta_true rad')
plt.ylabel('theta_est rad')
plt.title(f'{angle_type} = {angle_value} phi = [0, 4, 8, ..., 360]')
plt.grid(True)
save_plot()

# Scatter plot: parallel_error vs theta
plt.figure(figsize=(8, 5))
plt.scatter(results['phi_true'], results['parallel_error'], color='blue', alpha=0.6)
plt.xlabel('phi [rad]')
plt.ylabel('Parallel Error [nm]')
plt.title(f'{angle_type} = {angle_value} phi = [0, 4, 8, ..., 360]')
plt.grid(True)
save_plot()

# Scatter plot: perpendicular_error vs theta
plt.figure(figsize=(8, 5))
plt.scatter(results['phi_true'], results['perpendicular_error'], color='red', alpha=0.6)
plt.xlabel('phi [rad]')
plt.ylabel('Perpendicular Error [nm]')
plt.title(f'{angle_type} = {angle_value} phi = [0, 4, 8, ..., 360]')
plt.grid(True)
save_plot()

# Scatter plot: perpendicular_error vs parallel_error
plt.figure(figsize=(8, 5))
plt.scatter(results['parallel_error'], results['perpendicular_error'], color='purple', alpha=0.6)
plt.xlabel('Parallel Error [nm]')
plt.ylabel('Perpendicular Error [nm]')
plt.title(f'{angle_type} = {angle_value} phi = [0, 4, 8, ..., 360]')
plt.grid(True)
save_plot()

# Function to annotate mean and std deviation on histograms
def annotate_stats(data, color='black'):
    mean_val = np.mean(data)
    std_val = np.std(data)
    
    plt.axvline(mean_val, color='black', linestyle='dashed', linewidth=2, label=f'Mean = {mean_val:.2f}')
    plt.axvline(mean_val + std_val, color='gray', linestyle='dotted', linewidth=2, label=f'Std Dev = {std_val:.2f}')
    plt.axvline(mean_val - std_val, color='gray', linestyle='dotted', linewidth=2)
    
    plt.legend()

# Histogram: Frequency vs parallel_error
plt.figure(figsize=(8, 5))
plt.hist(results['parallel_error'], bins=30, color='blue', alpha=0.7, edgecolor='black')
plt.xlabel('Parallel Error [nm]')
plt.ylabel('Counts')
plt.title(f'{angle_type} = {angle_value} phi = [0, 4, 8, ..., 360]')
plt.grid(True)
annotate_stats(results['parallel_error'])
save_plot()

# Histogram: Frequency vs perpendicular_error
plt.figure(figsize=(8, 5))
plt.hist(results['perpendicular_error'], bins=30, color='red', alpha=0.7, edgecolor='black')
plt.xlabel('Perpendicular Error [nm]')
plt.ylabel('Counts')
plt.title(f'{angle_type} = {angle_value} phi = [0, 4, 8, ..., 360]')
plt.grid(True)
annotate_stats(results['perpendicular_error'])
save_plot()

# Histogram: delta x error
plt.figure(figsize=(8, 5))
plt.hist(x_err, bins=30, color='blue', alpha=0.7, edgecolor='black')
plt.xlabel('x error [nm]')
plt.ylabel('Counts')
plt.title(f'{angle_type} = {angle_value} phi = [0, 4, 8, ..., 360]')
plt.grid(True)
annotate_stats(x_err)
save_plot()

# Histogram: delta y error
plt.figure(figsize=(8, 5))
plt.hist(y_err, bins=30, color='red', alpha=0.7, edgecolor='black')
plt.xlabel('y error [nm]')
plt.ylabel('Counts')
plt.title(f'{angle_type} = {angle_value} phi = [0, 4, 8, ..., 360]')
plt.grid(True)
annotate_stats(y_err)
save_plot()

# Histogram: delta theta error
plt.figure(figsize=(8, 5))
plt.hist(theta_err, bins=30, color='blue', alpha=0.7, edgecolor='black')
plt.xlabel('theta error [nm]')
plt.ylabel('Counts')
plt.title(f'{angle_type} = {angle_value} phi = [0, 4, 8, ..., 360]')
plt.grid(True)
annotate_stats(theta_err)
save_plot()

# Histogram: delta phi error
plt.figure(figsize=(8, 5))
plt.hist(phi_err, bins=30, color='red', alpha=0.7, edgecolor='black')
plt.xlabel('phi error [nm]')
plt.ylabel('Counts')
plt.title(f'{angle_type} = {angle_value} phi = [0, 4, 8, ..., 360]')
plt.grid(True)
annotate_stats(phi_err)
save_plot()