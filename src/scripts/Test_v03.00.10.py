# ----------------------Version----------------------
# v03.00.00

# ----------------------Updates----------------------
# 1. Removed filter_outliers_by_label function
# 2. Focus on bounding box visualization and statistics
# 3. Using bottom center point method for vertical distance calculation
# 4. Applied captures directory structure from v01.03.05.01
# 5. Maintained single visualization approach
# 6. Added index remapping functionality from v01.03.05.01
# 7. Simplified DataFrame with remapped indices

import os
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import pandas as pd
import re
###########################
# Required PyQt5 related imports
from PyQt5.QtWidgets import QTableWidgetItem
from PyQt5.QtGui import QColor, QBrush, QFont
from PyQt5.QtCore import Qt

PIXEL_TO_MM_SCALE = 0.00459


def calculate_vertical_center_distances(boxes, labels, image_index=0):
    """
    Function to calculate vertical distances between bounding boxes and assign categories
    
    Parameters:
    - boxes: List of bounding boxes
    - labels: Corresponding label list
    - image_index: Sequential index of current image
    
    Returns:
    - DataFrame containing vertical distance measurements and categories
    """
    data = []
    
    # Return empty DataFrame if less than 2 boxes
    if len(boxes) < 2:
        return pd.DataFrame(columns=["Vertical_Distance", "Scaled_Distance", "Remapped_Index", "Category"])
    
    # Add information to bounding boxes (box, label, original index)
    objects = [(box, labels[i], i) for i, box in enumerate(boxes)]
    
    # Sort objects from top to bottom based on y_min
    objects.sort(key=lambda x: x[0][1])
    
    # Index mapping table - defines each segment of the die
    index_mappings = [
        [0, 1, 2],      # First image
        [2, 3, 4],      # Second image
        [4, 5, 6],      # Third image
        [6, 7, 8],      # Fourth image
        [8, 9, 10],     # Fifth image
        [10, 11, 12],   # Sixth image
        [12, 13, 14],   # Seventh image
        [14, 15, 16],   # Eighth image
        [16, 17]        # Ninth image
    ]
    
    # Define category mapping for all index transitions
    # Assign categories (E, F, G) for each index transition (0->1, 1->2, etc.)
    all_transition_categories = {
        "0-1": "E01",
        "1-2": "F02",
        "2-3": "G03",
        "3-4": "G04",
        "4-5": "G05",
        "5-6": "G06",
        "6-7": "G07",
        "7-8": "G08",
        "8-9": "G09",
        "9-10": "G10",
        "10-11": "G11",
        "11-12": "G12",
        "12-13": "G13",
        "13-14": "G14",
        "14-15": "G15",
        "15-16": "F16",
        "16-17": "E17"
    }
    
    # Get remapping for this image
    remapped_indices = []
    if 0 <= image_index < len(index_mappings):
        mapping = index_mappings[image_index]
        for i in range(min(len(objects), len(mapping))):
            remapped_indices.append(mapping[i])
    else:
        # Default to sequential indices if image_index is out of range
        remapped_indices = list(range(len(objects)))
    
    # Calculate distances between adjacent objects
    for i in range(len(objects) - 1):
        box1, label1, _ = objects[i]    # Upper object
        box2, label2, _ = objects[i + 1]  # Lower object
        
        # Skip if either label is 0 (background or invalid)
        if label1 == 0 or label2 == 0:
            continue
        
        # Calculate bottom center point of first box
        box1_bottom_center_x = (box1[0] + box1[2]) / 2  # x center
        box1_bottom_y = box1[3]  # y_max (bottom of box)
        
        # Determine vertical distance calculation method
        if box2[0] <= box1_bottom_center_x <= box2[2]:
            # Center point is within x range of second box
            box2_top_y = box2[1]  # y_min (top of box)
            vertical_distance = box2_top_y - box1_bottom_y
        else:
            # Center point is outside x range of second box
            if box1_bottom_center_x < box2[0]:
                # Center is to the left of second box
                vertical_distance = np.sqrt(
                    (box2[0] - box1_bottom_center_x)**2 + (box2[1] - box1_bottom_y)**2
                )
            else:
                # Center is to the right of second box
                vertical_distance = np.sqrt(
                    (box1_bottom_center_x - box2[2])**2 + (box2[1] - box1_bottom_y)**2
                )
        
        # Scale distance (convert from pixels to mm)
        scaled_distance = vertical_distance * PIXEL_TO_MM_SCALE
        
        # Generate remapped index string
        if i < len(remapped_indices) and i+1 < len(remapped_indices):
            from_idx = remapped_indices[i]
            to_idx = remapped_indices[i+1]
            remapped_idx_str = f"{from_idx}->{to_idx}"
            
            # Generate transition key (e.g., "0-1", "1-2", etc.)
            transition_key = f"{from_idx}-{to_idx}"
            
            # Get category corresponding to transition
            if transition_key in all_transition_categories:
                category = all_transition_categories[transition_key]
            else:
                # Default to G for unknown transitions
                category = 'G'
        else:
            # If no remapped indices available
            remapped_idx_str = f"{i}->{i+1}"
            # Assign default categories
            if i == 0:
                category = 'E'
            elif i == len(objects) - 2:
                category = 'E'
            elif i == 1 or i == len(objects) - 3:
                category = 'F'
            else:
                category = 'G'
        
        # Add distance data
        data.append([
            vertical_distance,
            scaled_distance,
            remapped_idx_str,
            category
        ])
    
    # Create and return DataFrame
    return pd.DataFrame(data, columns=["Vertical_Distance", "Scaled_Distance", "Remapped_Index", "Category"])

def test_and_visualize(model, input_dir, output_dir, threshold=0.5):
    """
    Image testing and result visualization function (simplified version)
    
    Parameters:
    model: Object detection model
    input_dir: Input image directory
    output_dir: Result save directory
    threshold: Object detection threshold
    
    Returns:
    DataFrame: Dataframe containing distance calculation results for all images
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.eval()
    model.to(device)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    all_distances = []
    
    # Get image files from input directory
    allowed_patterns = ["capture_1.png", "capture_2.png", "capture_3.png", 
                       "capture_4.png", "capture_5.png", "capture_6.png", 
                       "capture_7.png", "capture_8.png", "capture_9.png"]
    
    image_files = []
    for pattern in allowed_patterns:
        path = os.path.join(input_dir, pattern)
        if os.path.exists(path):
            image_files.append(path)
        else:
            print(f"Warning: File not found - {path}")
    
    if not image_files:
        print(f"Error: No image files to process.")
        return pd.DataFrame()  # Return empty dataframe

    for idx, img_path in enumerate(image_files):
        print(f"Processing image {idx+1}/{len(image_files)}: {os.path.basename(img_path)}")
        
        # Extract image number
        img_filename = os.path.basename(img_path)
        match = re.search(r'capture_(\d+)', img_filename)
        if match:
            img_number = int(match.group(1)) - 1  # 0-based index
        else:
            img_number = idx % 9  # Default value
        
        # Load image and predict
        original_img = Image.open(img_path).convert('RGB')
        original_np = np.array(original_img)
        img_tensor = transforms.ToTensor()(original_np).to(device)

        with torch.no_grad():
            prediction = model([img_tensor])

        # Process prediction results
        boxes = [box.cpu().numpy() for i, box in enumerate(prediction[0]['boxes']) 
                if prediction[0]['scores'][i] > threshold]
        labels = [int(prediction[0]['labels'][i].cpu().numpy()) for i, score 
                in enumerate(prediction[0]['scores']) if score > threshold]
        scores = [float(score.cpu().numpy()) for i, score 
                in enumerate(prediction[0]['scores']) if score > threshold]
        
        # Calculate distances
        df_distances = calculate_vertical_center_distances(boxes, labels, img_number)
        all_distances.append(df_distances)
        
        # Visualization (for GUI viewing)
        result_img = Image.fromarray(original_np.copy())
        draw = ImageDraw.Draw(result_img)
        
        for i, box in enumerate(boxes):
            label = labels[i]
            color = "green" if label == 1 else "blue" if label == 2 else "red" if label == 3 else "yellow"
            draw.rectangle(box.tolist(), outline=color, width=3)
            
            # Display label and score
            score_text = f"{scores[i]:.2f}" if i < len(scores) else "N/A"
            draw.text((box[0], box[1] - 10), f"{label}", fill=color)
            draw.text((box[0], box[1] - 20), f"{score_text}", fill=color)
            
            # Display center point
            center_x = (box[0] + box[2]) / 2
            bottom_y = box[3]
            draw.ellipse((center_x-3, bottom_y-3, center_x+3, bottom_y+3), fill="white", outline="black")
        
        # Save result image
        result_path = os.path.join(output_dir, "visualizations", f"{img_filename}")
        result_img.save(result_path)
        print(f"  Result image saved: {result_path}")
   
    # Combine all distance calculation results into one dataframe
    if all_distances:
        return pd.concat(all_distances, ignore_index=True)
    else:
        return pd.DataFrame()  # Return empty dataframe
    
    
def load_model(model_path):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found - {model_path}")
        exit(1)
    
    # Load state dictionary
    try:
        state_dict = torch.load(model_path, map_location=device)
        print(f"Model file loaded: {model_path}")
    except Exception as e:
        print(f"Error: Failed to load model file: {e}")
        exit(1)
    
    # Determine number of classes
    cls_score_weight = state_dict.get('roi_heads.box_predictor.cls_score.weight')
    if cls_score_weight is not None:
        num_classes = cls_score_weight.size(0)
    else:
        num_classes = 4  # Default value
    
    # Create model (simplified: use only Faster R-CNN)
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    model = fasterrcnn_resnet50_fpn(weights=None)
    
    # Update box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Load state dictionary
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"Note: Some parameters were not loaded.")
    
    model.to(device)
    return model


# Main execution code
if __name__ == "__main__":
    print("Running test code v03.00.05 - Object detection, vertical distance calculation and index remapping (9 image support)")
    
    try:
        # Set base path
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Get paths and settings from environment variables
        input_dir = os.environ.get('INPUT_DIR')
        output_dir = os.environ.get('OUTPUT_DIR')
        model_path = os.environ.get('MODEL_PATH')
        
        # Set default values if paths are not available
        if not input_dir or not os.path.exists(input_dir):
            input_dir = os.path.join(BASE_DIR, "captures")
            print(f"Using default input directory: {input_dir}")
        
        if not output_dir:
            output_dir = os.path.join(BASE_DIR, "results")
            print(f"Using default output directory: {output_dir}")
        
        if not model_path or not os.path.exists(model_path):
            model_path = os.path.join(BASE_DIR, "models", "model_only.pth")
            print(f"Using default model path: {model_path}")
        
        # Check and create directories
        if not os.path.exists(input_dir):
            print(f"Error: Input directory does not exist - {input_dir}")
            exit(1)
            
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
        
        # Clean existing visualization files
        clean_visualizations = True  # Can be set to False if needed
        if clean_visualizations:
            visualizations_dir = os.path.join(output_dir, "visualizations")
            for file in os.listdir(visualizations_dir):
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        os.remove(os.path.join(visualizations_dir, file))
                    except Exception as e:
                        print(f"Error deleting file: {e}")
        
        # Output configuration information
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Model path: {model_path}")

        # Load model
        model = load_model(model_path)
        
        # Execute test and visualization
        test_and_visualize(
            model, 
            input_dir, 
            output_dir, 
            threshold=0.3,
        )
        
        print("Test completed!")
        
    except Exception as e:
        import traceback
        print(f"Error occurred during test execution: {e}")
        traceback.print_exc()