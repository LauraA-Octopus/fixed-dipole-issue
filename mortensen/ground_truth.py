import cv2
import numpy as np

def ground_truth_with_drawing(image_path, output_path, grid_size=(7, 7), spacing=10):
    """
    Compute the ground truth centroids of a grid of dipoles and draw them on the image.

    Parameters:
        image_path (str): Path to the fixed image containing the dipoles.
        output_path (str): Path to save the image with drawn centroids.
        grid_size (tuple): Number of rows and columns in the dipole grid.
        spacing (int): Pixel distance between adjacent dipoles in the grid.

    Returns:
        centroids (list): List of (x, y) tuples representing the centroid locations.
    """
    # Load the fixed image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Determine the center of the image
    height, width = image.shape
    center_x, center_y = width // 2, height // 2

    # Generate grid of centroids around the center
    centroids = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            x = center_x + (j - (grid_size[1] - 1) // 2) * spacing
            y = (center_y + (i - (grid_size[0] - 1) // 2) * spacing) - 2
            centroids.append((x, y))

            # Draw a red dot at the centroid location
            cv2.circle(image, (x, y), radius=1, color=(255, 0, 0), thickness=-1)

    # Save the image with drawn centroids
    cv2.imwrite(output_path, image)
    print(f"Image with centroids saved to {output_path}")

    return centroids

def find_centroids(image_path, output_path, threshold=15, min_area=4):
    """
    Automatically detect centroids of bright regions (dipoles) in the image.

    Parameters:
        image_path (str): Path to the fixed image containing the dipoles.
        output_path (str): Path to save the image with drawn centroids.
        threshold (int): Pixel intensity threshold for binarization.
        min_area (int): Minimum area of detected regions to consider as dipoles.

    Returns:
        centroids (list): List of (x, y) tuples representing the detected centroid locations.
    """
    # Load the fixed image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Threshold the image to create a binary mask
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            # Calculate the centroid of the contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))

                # Draw the centroid on the image
                cv2.circle(image, (cx, cy), radius=1, color=(0, 255, 0), thickness=-1)

    # Save the image with detected centroids
    cv2.imwrite(output_path, image)
    print(f"Image with detected centroids saved to {output_path}")

    return centroids

def compare_centroids(image_path, ground_truth, detected, output_path):
    """
    Draw both ground truth and detected centroids on the same image for comparison.

    Parameters:
        image_path (str): Path to the fixed image containing the dipoles.
        ground_truth (list): Ground truth centroids as a list of (x, y) tuples.
        detected (list): Detected centroids as a list of (x, y) tuples.
        output_path (str): Path to save the image with drawn centroids.
    """
    # Load the fixed image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Convert to color image for better visualization
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw ground truth centroids (red)
    for x, y in ground_truth:
        cv2.circle(image, (x, y), radius=1, color=(0, 0, 255), thickness=-1)

    # Draw corrected detected centroids (green)
    for x, y in detected:
        cv2.circle(image, (x, y), radius=1, color=(0, 255, 0), thickness=-1)

    # Save the comparison image
    cv2.imwrite(output_path, image)
    print(f"Comparison image saved to {output_path}")
    
def draw_patches(image_path, centroids, output_path, patch_width=12):
    """
    Draw patches around the given centroids on the image.

    Parameters:
        image_path (str): Path to the image file.
        centroids (list): List of (x, y) tuples representing the centroids.
        output_path (str): Path to save the image with patches drawn.
        patch_width (int): Width of the square patches.
    """
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Convert to color image for better visualization
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for cx, cy in centroids:
        # Define the patch coordinates
        x_start = max(0, cx - patch_width // 2)
        x_end = min(image.shape[1], cx + patch_width // 2)
        y_start = max(0, cy - patch_width // 2)
        y_end = min(image.shape[0], cy + patch_width // 2)

        # Draw a rectangle around the patch
        cv2.rectangle(image, (x_start, y_start), (x_end, y_end), color=(255, 255, 0), thickness=1)

    # Save the image with patches
    cv2.imwrite(output_path, image)
    print(f"Image with patches saved to {output_path}")

# Example usage for ground truth
image_path = r"C:\Users\laura\fixed-dipole-issue\mortensen\image_stack_fixdip_maxproject.png"
output_path_ground_truth = r"C:\Users\laura\fixed-dipole-issue\mortensen\image_with_dipoles.png"
dipole_centroids = ground_truth_with_drawing(image_path, output_path_ground_truth)
print(f"Ground truth centroids: {dipole_centroids}")

# Example usage for finding centroids
default_output_path_detected = r"C:\Users\laura\fixed-dipole-issue\mortensen\image_with_detected_centroids.png"
detected_centroids = find_centroids(image_path, default_output_path_detected)
print(f"Detected centroids: {detected_centroids}")

# Compare ground truth and detected centroids
comparison_output_path = r"C:\Users\laura\fixed-dipole-issue\mortensen\comparison_image.png"
compare_centroids(image_path, dipole_centroids, detected_centroids, comparison_output_path)

# Draw patches around detected centroids
patches_output_path = r"C:\Users\laura\fixed-dipole-issue\mortensen\image_with_patches.png"
draw_patches(image_path, detected_centroids, patches_output_path)
