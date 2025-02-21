import csv
import os

def save_results_to_csv(results, ground_truth, filename="results.csv"):
    """
    Appends the results and ground truth to a CSV file.
    
    Parameters:
    - results: Tuple of estimated values (phi, theta, x, y, photons)
    - ground_truth: Tuple of ground truth values (phi, theta, x, y, photons)
    - filename: CSV filename (default: "results.csv")
    """

    file_exists = os.path.isfile(filename)

    # Open the CSV file in append mode
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write header if the file does not exist
        if not file_exists:
            writer.writerow(["phi_est", "theta_est", "x_est", "y_est", "photons_est",
                             "phi_true", "theta_true", "x_true", "y_true", "photons_true"])

        # Write the results
        writer.writerow(list(results) + list(ground_truth))