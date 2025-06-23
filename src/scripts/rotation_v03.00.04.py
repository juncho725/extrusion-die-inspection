# GUI Rotation code

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os
from pathlib import Path

#######################################################
# Preprocessing functions
#######################################################

def enhance_image(gray_image):
    """
    Enhance image contrast and sharpness
    """
    # Contrast enhancement with CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_image)
    
    # Sharpness enhancement (optional)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    return sharpened

def remove_red_lines(image):
    """
    Remove red measurement lines
    """
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define red color range (red in HSV is 0-10 degrees and 160-180 degrees)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Combine both masks
    red_mask = mask1 + mask2
    
    # Invert mask (everything except red)
    red_mask_inv = cv2.bitwise_not(red_mask)
    
    # Apply mask to original image (remove red parts)
    result = cv2.bitwise_and(image, image, mask=red_mask_inv)
    
    return result

def create_side_mask(image, exclude_percentage=25, smooth_border=True):
    """
    Create mask that excludes specified percentage from both sides of image
    
    Parameters:
    - image: Input image
    - exclude_percentage: Percentage to exclude from both sides (%)
    - smooth_border: Whether to smooth the border
    
    Returns:
    - mask: Mask showing valid area (center part is 255, sides are 0)
    """
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Calculate pixels to exclude
    exclude_pixels = int(width * (exclude_percentage / 100))
    
    if smooth_border:
        # Gradient area width (pixels)
        gradient_width = int(width * 0.05)  # Set 5% of total width as gradient
        
        # Set center area (fully included)
        mask[:, exclude_pixels+gradient_width:width-(exclude_pixels+gradient_width)] = 255
        
        # Left border gradient
        for i in range(gradient_width):
            # Linear increase from 0 to 255
            value = int((i / gradient_width) * 255)
            mask[:, exclude_pixels+i] = value
            
        # Right border gradient
        for i in range(gradient_width):
            # Linear decrease from 255 to 0
            value = int(((gradient_width - i) / gradient_width) * 255)
            mask[:, width-exclude_pixels-gradient_width+i] = value
    else:
        # Only center area in mask without gradient (excluding both sides)
        mask[:, exclude_pixels:width-exclude_pixels] = 255
    
    return mask

def create_masked_binary(image_path, exclude_sides_percent=25, smooth_border=True):
    """
    Preprocess image to create masked binary image
    
    Parameters:
    - image_path: Image file path
    - exclude_sides_percent: Percentage of area to exclude from both sides (%)
    - smooth_border: Whether to smooth mask borders
    
    Returns:
    - original: Original image
    - masked_binary: Preprocessed binary image
    """
    # Load image
    original = cv2.imread(image_path)
    if original is None:
        print(f"Cannot load image: {image_path}")
        return None, None
    
    # Remove red measurement lines
    no_red_lines = remove_red_lines(original)
    
    # Convert to grayscale
    gray = cv2.cvtColor(no_red_lines, cv2.COLOR_BGR2GRAY)
    
    # Enhance image
    enhanced = enhance_image(gray)
    
    # Gaussian blur for noise reduction
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Binarization (using Otsu algorithm)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological operations for noise removal and region connection
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Create mask excluding both sides
    side_mask = create_side_mask(morph, exclude_sides_percent, smooth_border)
    
    # Apply mask to exclude side areas
    masked_morph = cv2.bitwise_and(morph, morph, mask=side_mask)
    
    # Apply morphological operations to prevent artificial contours at borders
    if smooth_border:
        # Light erosion to remove edge noise
        erode_kernel = np.ones((3, 3), np.uint8)
        masked_morph = cv2.erode(masked_morph, erode_kernel, iterations=1)
        
        # Dilation to restore eroded parts (noise already removed)
        dilate_kernel = np.ones((3, 3), np.uint8)
        masked_morph = cv2.dilate(masked_morph, dilate_kernel, iterations=1)
    
    return original, masked_morph

#######################################################
# Angle calculation functions (left and right edge processing)
#######################################################

def detect_left_edge(binary_image):
    """
    Detect left edge contour of object
    
    Parameters:
    - binary_image: Binary image (white object on black background)
    
    Returns:
    - left_edge_points: List of points representing left edge
    """
    # Find leftmost white pixel in each row
    height, width = binary_image.shape
    left_edge_points = []
    
    for y in range(height):
        # Find first white pixel from left to right
        row = binary_image[y, :]
        white_pixels = np.where(row > 0)[0]
        
        if len(white_pixels) > 0:
            left_x = white_pixels[0]
            left_edge_points.append((left_x, y))
    
    return left_edge_points

def detect_right_edge(binary_image):
    """
    Detect right edge contour of object
    
    Parameters:
    - binary_image: Binary image (white object on black background)
    
    Returns:
    - right_edge_points: List of points representing right edge
    """
    # Find rightmost white pixel in each row
    height, width = binary_image.shape
    right_edge_points = []
    
    for y in range(height):
        # Find last white pixel from left to right
        row = binary_image[y, :]
        white_pixels = np.where(row > 0)[0]
        
        if len(white_pixels) > 0:
            right_x = white_pixels[-1]  # Rightmost pixel
            right_edge_points.append((right_x, y))
    
    return right_edge_points

def find_density_regions(edge_points, bin_width=5, top_n=2):
    """
    Find regions with highest density of points based on x coordinates
    
    Parameters:
    - edge_points: List of (x, y) points
    - bin_width: Histogram bin width
    - top_n: Number of highest density regions to return
    
    Returns:
    - dense_regions: List of (center_x, count) tuples for highest density regions
    """
    if not edge_points:
        return []
    
    # Extract x coordinates
    x_coords = [p[0] for p in edge_points]
    
    # Create histogram of x coordinates
    min_x, max_x = min(x_coords), max(x_coords)
    bins = np.arange(min_x, max_x + bin_width, bin_width)
    hist, bin_edges = np.histogram(x_coords, bins=bins)
    
    # Get bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Create (center_x, count) pairs and sort by count in descending order
    density_regions = [(center, count) for center, count in zip(bin_centers, hist)]
    density_regions.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N density regions
    return density_regions[:top_n]

def get_points_in_region(edge_points, center_x, tolerance):
    """
    Get points within specified tolerance from center x coordinate
    
    Parameters:
    - edge_points: List of (x, y) points
    - center_x: Center x coordinate
    - tolerance: Tolerance distance from center x
    
    Returns:
    - region_points: List of points within specified region
    """
    region_points = [(x, y) for x, y in edge_points if abs(x - center_x) <= tolerance]
    return region_points

def fit_line_to_edge(edge_points):
    """
    Fit line to edge points using linear regression
    
    Parameters:
    - edge_points: List of (x, y) points representing edge
    
    Returns:
    - slope: Slope of fitted line
    - intercept: Y-intercept of fitted line
    - r_value: Correlation coefficient
    """
    if not edge_points or len(edge_points) < 2:
        return None, None, None
    
    x_coords = [p[0] for p in edge_points]
    y_coords = [p[1] for p in edge_points]
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(y_coords, x_coords)
    
    return slope, intercept, r_value

def calculate_angle_from_vertical(slope):
    """
    Calculate angle from vertical based on slope
    
    Parameters:
    - slope: Slope of fitted line
    
    Returns:
    - angle: Angle from vertical (in degrees)
    """
    if slope is None:
        return None
    
    # Calculate angle from vertical (arctangent of slope)
    angle = np.degrees(np.arctan(slope))
    
    return angle

def calculate_rotation_angle(binary_image, bin_width=5, tolerance=5):
    """
    Calculate object rotation angle from binary image (considering both left and right edges)
    
    Parameters:
    - binary_image: Binary image
    - bin_width: Histogram bin width
    - tolerance: Tolerance distance from center x
    
    Returns:
    - angle: Angle from vertical (in degrees)
    - visualization: Visualization image (optional)
    """
    # Detect left edge and calculate angle
    left_edge_points = detect_left_edge(binary_image)
    left_dense_regions = find_density_regions(left_edge_points, bin_width=bin_width)
    
    left_angle = None
    if left_dense_regions:
        center_x, _ = left_dense_regions[0]
        region_points = get_points_in_region(left_edge_points, center_x, tolerance)
        slope, intercept, r_value = fit_line_to_edge(region_points)
        left_angle = calculate_angle_from_vertical(slope)
    
    # Detect right edge and calculate angle
    right_edge_points = detect_right_edge(binary_image)
    right_dense_regions = find_density_regions(right_edge_points, bin_width=bin_width)
    
    right_angle = None
    if right_dense_regions:
        center_x, _ = right_dense_regions[0]
        region_points = get_points_in_region(right_edge_points, center_x, tolerance)
        slope, intercept, r_value = fit_line_to_edge(region_points)
        right_angle = calculate_angle_from_vertical(slope)
    
    # Determine angle: use average of both angles or valid single angle
    final_angle = None
    if left_angle is not None and right_angle is not None:
        # Calculate average of both angles
        final_angle = (left_angle + right_angle) / 2
        print(f"Left angle: {left_angle:.2f}°, Right angle: {right_angle:.2f}°, Average: {final_angle:.2f}°")
    elif left_angle is not None:
        final_angle = left_angle
        print(f"Using left angle only: {final_angle:.2f}°")
    elif right_angle is not None:
        final_angle = right_angle
        print(f"Using right angle only: {final_angle:.2f}°")
    else:
        print("Angle calculation failed: Cannot find valid edges on both left and right sides.")
    
    return final_angle, None

#######################################################
# Image rotation function
#######################################################

def rotate_image(image, angle):
    """
    Rotate image by given angle (to make object vertical)
    
    Parameters:
    - image: Image to rotate
    - angle: Angle from vertical (in degrees)
    
    Returns:
    - rotated_image: Rotated image
    """
    if angle is None:
        return image
    
    # Calculate image center point
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    # Calculate rotation transformation matrix (rotate in opposite direction to make object vertical)
    # Reverse angle sign to make object vertical
    rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
    
    # Apply rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)
    
    return rotated_image

#######################################################
# Main processing function (modified for GUI integration)
#######################################################

def process_and_rotate_image(image_path, bin_width=5, tolerance=5, exclude_sides_percent=25, smooth_border=True):
    """
    Process image, calculate angle, apply rotation and overwrite original file
    
    Parameters:
    - image_path: Image file path
    - bin_width: Histogram bin width
    - tolerance: Tolerance distance from center x
    - exclude_sides_percent: Percentage of area to exclude from both sides (%)
    - smooth_border: Whether to smooth mask borders
    
    Returns:
    - angle: Calculated rotation angle
    """
    # Extract filename (excluding path)
    image_name = os.path.basename(image_path)
    
    # 1. Create masked binary image through preprocessing
    original, masked_binary = create_masked_binary(
        image_path, 
        exclude_sides_percent=exclude_sides_percent,
        smooth_border=smooth_border
    )
    
    if original is None or masked_binary is None:
        print(f"Image processing failed: {image_path}")
        return None
    
    # 2. Calculate angle considering both edges
    angle, _ = calculate_rotation_angle(
        masked_binary,
        bin_width=bin_width,
        tolerance=tolerance
    )
    
    if angle is None:
        print(f"Angle calculation failed: {image_path}")
        return None
    
    print(f"[{image_name}] Rotation angle: {angle:.2f}°")
    
    # 3. Apply rotation
    rotated_image = rotate_image(original, angle)
    
    # 4. Overwrite original file
    cv2.imwrite(image_path, rotated_image)
    print(f"File overwrite complete: {image_path}")
    
    return angle

def rotate_all_images_in_directory(directory_path, bin_width=5, tolerance=5, exclude_sides_percent=25, smooth_border=True):
    """
    Process and rotate all images in specified directory (overwrite original files)
    
    Parameters:
    - directory_path: Directory path containing image files
    - bin_width: Histogram bin width
    - tolerance: Tolerance distance from center x
    - exclude_sides_percent: Percentage of area to exclude from both sides (%)
    - smooth_border: Whether to smooth mask borders
    
    Returns:
    - results: List of processed images and angle information
    """
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return []
    
    # Find image files in directory
    image_files = []
    valid_extensions = ['.png', '.jpg', '.jpeg']
    for ext in valid_extensions:
        image_files.extend(list(Path(directory_path).glob(f"*{ext}")))
    
    if not image_files:
        print(f"No image files in directory: {directory_path}")
        return []
    
    results = []
    for i, img_path in enumerate(image_files):
        img_path_str = str(img_path)
        print(f"\n[{i+1}/{len(image_files)}] Processing image: {os.path.basename(img_path_str)}")
        
        angle = process_and_rotate_image(
            img_path_str,
            bin_width=bin_width,
            tolerance=tolerance,
            exclude_sides_percent=exclude_sides_percent,
            smooth_border=smooth_border
        )
        
        results.append((img_path_str, angle))
    
    print(f"\nAll images processed. Total {len(results)} images processed.")
    return results

# Function for processing specific 9 capture images (for GUI integration)
def rotate_capture_set(capture_dir, bin_width=5, tolerance=5, exclude_sides_percent=25, smooth_border=True):
    """
    Find and rotate capture_1.png ~ capture_9.png files in capture directory
    
    Parameters:
    - capture_dir: Directory path where capture images are stored
    - bin_width: Histogram bin width
    - tolerance: Tolerance distance from center x
    - exclude_sides_percent: Percentage of area to exclude from both sides (%)
    - smooth_border: Whether to smooth mask borders
    
    Returns:
    - results: List of processed images and angle information
    """
    if not os.path.exists(capture_dir):
        print(f"Capture directory not found: {capture_dir}")
        return []
    
    results = []
    # Process capture_1.png ~ capture_9.png
    for i in range(1, 10):
        capture_file = os.path.join(capture_dir, f"capture_{i}.png")
        if os.path.exists(capture_file):
            print(f"\n[{i}/9] Processing image: capture_{i}.png")
            
            angle = process_and_rotate_image(
                capture_file,
                bin_width=bin_width,
                tolerance=tolerance,
                exclude_sides_percent=exclude_sides_percent,
                smooth_border=smooth_border
            )
            
            results.append((capture_file, angle))
        else:
            print(f"File not found: {capture_file}")
    
    print(f"\nRotation processing complete. Total {len(results)} images processed.")
    return results

# Main function for test execution
if __name__ == "__main__":
    import sys
    
    # Get directory path from command line arguments
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
        print(f"Starting image processing for specified directory: {target_dir}")
        rotate_all_images_in_directory(target_dir)
    else:
        # Check directory path from environment variables
        input_dir = os.environ.get("INPUT_DIR")
        if input_dir:
            print(f"Starting image processing for directory specified by environment variable: {input_dir}")
            rotate_capture_set(input_dir)
        else:
            print("Please specify directory path to process.")
            print("Usage: python rotation_script.py [directory_path]")
            print("Or set environment variable INPUT_DIR.")