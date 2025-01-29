import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the folder containing the CSV files
folder = os.path.expanduser('~/Documents/Hinterer/fixed-dipole-issue/mortensen/simulation_results/parallel_perpendicular_errors')
base_filename = "parallel_perpendicular_errors_"

# Initialize the plot
plt.figure(figsize=(12, 8))
plt.title("Perpendicular Error as a Function of Dipole (All Datasets)")
plt.xlabel("Dipole")
plt.ylabel("Perpendicular Error")
plt.grid(True)

# Loop through all 81 files
for i in range(1, 79):  # Files are named 1 to 81
    file_path = os.path.join(folder, f"{base_filename}{i}.csv")
    if os.path.exists(file_path):
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Plot the parallel_error as a function of Dipole
        plt.scatter(df['Dipole'], df['perpendicular_error'], color='blue', alpha=0.7, s=10)


plt.tight_layout()

# Show the plot
plt.show()