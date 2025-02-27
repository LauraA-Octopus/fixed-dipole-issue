import subprocess
import numpy as np
from save_results import save_results_to_csv

def run_test(phi, theta, filename, num_repeats=3):
    """
    Runs test.py with given phi and theta values, num_repeats times, and saves results to a specified CSV file.
    """
    for _ in range(num_repeats):
        process = subprocess.run(["python", "test.py", str(phi), str(theta)], capture_output=True, text=True)
        
        # Extract results from test.py output
        output_lines = process.stdout.strip().split("\n")
        result_line = [line for line in output_lines if "Results from the Mortensen fit are:" in line]
        gt_line = [line for line in output_lines if "Ground truth are:" in line]
        
        if result_line and gt_line:
            try:
                # Extract estimated results
                result_str = result_line[0].split(":")[1].strip()
                results = result_str #np.fromstring(result_str.strip('[]'), sep=' ').tolist()
                print(f"Mortensen fitting results are: {results}")
                
                # Extract ground truth values
                ground_truth_str = gt_line[0].split(":")[1].strip()
                ground_truth = list(map(float, ground_truth_str.split(",")))
                print(f"Extracted ground truth string: {ground_truth}")
                
                # Save results
                save_results_to_csv(results, ground_truth, filename)

            except ValueError as e:
                print(f"Error parsing results: {e}")
                continue

def main():
    # Define theta and phi values
    chosen_thetas = [0, 22.5, 45, 67.5, 90]
    chosen_phis = [0, 45, 135, 180, 225, 270, 315]
    fixed_phis = list(range(0, 361, 4))  # [0, 4, 8, ..., 360]
    fixed_thetas = list(range(0, 91, 1))
    num_repeats = 3

    # Run tests for each theta and varying phi values
    for theta in chosen_thetas:
        filename = f"results_theta_{theta}.csv"
        for phi in fixed_phis:
            run_test(phi * np.pi / 180, theta * np.pi / 180, filename, num_repeats)

    for phi in chosen_phis:
        filename = f"results_phi_{phi}.csv"
        for theta in fixed_thetas:
            run_test(phi * np.pi / 180, theta * np.pi / 180, filename, num_repeats)

if __name__ == "__main__":
    main()


    
    # Run tests for fixed theta values with varying phi values
    #for theta in fixed_thetas:
    #    filename = f"results_theta_{theta}.csv"
    #    for i, phi in enumerate(fixed_phis):
    #        # Change phi after every 'num_copies' (3) iterations
    #        if i % common_repeats == 0 and i > 0:
    #            phi += 4  # Increment phi by 4 degrees after every 20 simulations
    #        run_test(phi * np.pi / 180, theta * np.pi / 180, filename, common_repeats)
    
    # Run tests for fixed phi values with varying theta values
    #for phi in fixed_phis:
    #    filename = f"results_phi_{phi}.csv"
    #    for theta in [0, 22.5, 45, 67.5, 90]:
    #        run_test(phi * np.pi / 180, theta * np.pi / 180, filename, common_repeats)
