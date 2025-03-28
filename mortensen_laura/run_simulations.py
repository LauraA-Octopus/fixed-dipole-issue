import numpy as np
import os
import test
from save_results import save_results_to_csv

def run_test(phi, theta, filename, num_repeats=15):
    """
    Runs the Mortensen fit from test.py directly via function call.
    """
    for _ in range(num_repeats):
        try:
            results, ground_truth = test.run_mortensen_fit(phi, theta)

            results = list(map(float, results))  
            ground_truth = list(map(float, ground_truth))  

            print(f"Mortensen fitting results are: {results}")
            print(f"Extracted ground truth string: {ground_truth}")

            # Save results
            save_results_to_csv(results, ground_truth, filename)

        except ValueError as e:
            print(f"Error parsing results: {e}")
            continue

def main():

    # Define the output folder
    output_folder = "/home/wgq72938/Documents/Hinterer/fixed-dipole-issue/mortensen_laura/plots/test_5repsFit_280325"
    os.makedirs(output_folder, exist_ok=True) 
    # Define theta and phi values
    chosen_thetas = [0, 22.5, 45, 67.5, 90]
    chosen_phis = [0, 45, 135, 180, 225, 270, 315]
    fixed_phis = list(range(0, 361, 4))  
    fixed_thetas = list(range(0, 91, 2))
    num_repeats = 15

    # Run tests for each theta and varying phi values
    for theta in chosen_thetas:
        filename = os.path.join(output_folder, f"results_theta_{theta}.csv")
        for phi in fixed_phis:
            run_test(phi * np.pi / 180, theta * np.pi / 180, filename, num_repeats)

    for phi in chosen_phis:
        filename = f"results_phi_{phi}.csv"
        for theta in fixed_thetas:
            run_test(phi * np.pi / 180, theta * np.pi / 180, filename, num_repeats)

if __name__ == "__main__":
    main()