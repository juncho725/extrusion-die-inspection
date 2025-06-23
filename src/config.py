# -*- coding: utf-8 -*-
"""
Configuration file - Defines basic setting values for the program
"""

import os

# =============================================================================
# Path Settings
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 프로젝트 루트
base_folder = PROJECT_ROOT

photo_folder = os.path.join(base_folder, "Captures")
settings_folder = os.path.join(base_folder, "Settings")
results_folder = os.path.join(base_folder, "Results")

# Code file paths
rotation_code_path = os.path.join(PROJECT_ROOT, "src", "scripts", "rotation_v03.00.04.py")
test_code_path = os.path.join(PROJECT_ROOT, "src", "scripts", "Test_v03.00.10.py")
model_file_path = os.path.join(PROJECT_ROOT, "src", "models", "model_only.pth")

# Image file paths
company_logo_path = os.path.join(PROJECT_ROOT, "src", "resources", "images", "company_logo.png")
drawing_image_path = os.path.join(PROJECT_ROOT, "src", "resources", "images", "drawing.png")

# =============================================================================
# Capture Area Settings
# =============================================================================
capture_x_coord = 35
capture_y_coord = 150
capture_width = 640
capture_height = 480

# =============================================================================
# Default Value Settings
# =============================================================================
# Target distance values (unit: mm)
default_target_distance = {
    "A": 15.37,
    "B": 1.23,
    "E": 0.25,
    "F": 0.25,
    "G": 0.24
}

# Tolerance (+ direction, unit: mm)
default_tolerance_plus = {
    "A": 0.010,
    "B": 0.010,
    "E": 0.010,
    "F": 0.010,
    "G": 0.010
}

# Tolerance (- direction, unit: mm)
default_tolerance_minus = {
    "A": 0.010,
    "B": 0.010,
    "E": 0.010,
    "F": 0.010,
    "G": 0.010
}

# =============================================================================
# UI Style Settings
# =============================================================================
# Button style
button_style = """
QPushButton { 
    padding: 10px 15px; 
    font-size: 20px; 
    font-weight: 900; 
    font-family: 'Arial Black', 'Arial Bold', Arial; 
}
"""

# Menu button style
menu_button_style = """
QPushButton { 
    padding: 10px 15px; 
    font-size: 25px; 
    font-weight: 1800; 
    font-family: 'Arial Black', 'Arial Bold', Arial; 
}
"""

# Capture button style
capture_button_style = """
QPushButton { 
    background-color: #E74C3C; 
    color: white; 
    font-weight: 900; 
    font-size: 25px; 
    font-family: 'Arial Black', 'Arial Bold', Arial;
    border-radius: 12px;
    border: 2px solid #C0392B;
}
QPushButton:hover {
    background-color: #EC7063;
}
QPushButton:pressed {
    background-color: #C0392B;
}
"""

# AI analysis button style
ai_analysis_button_style = """
QPushButton { 
    background-color: #4CAF50; 
    color: white; 
    font-weight: 900; 
    font-size: 25px; 
    font-family: 'Arial Black', 'Arial Bold', Arial; 
}
"""

# Product navigation button style
product_nav_button_style = """
QPushButton { 
    background-color: #4a9cff; 
    color: white; 
    font-weight: 900; 
    font-size: 25px; 
    font-family: 'Arial Black', 'Arial Bold', Arial; 
}
"""

# =============================================================================
# Color Settings
# =============================================================================
tolerance_over_color = "#ffcccc"  # Red (tolerance exceeded)
tolerance_under_color = "#ccccff"  # Blue (tolerance under)
selected_column_color = "#f5f5f5"  # Light gray (selected column)
default_background_color = "#dcdcdc"  # Default gray

# =============================================================================
# Analysis Category Settings
# =============================================================================
analysis_category_list = ["A00", "B00", "E01", "F02"] + [f"G{i:02d}" for i in range(3, 16)] + ["F16", "E17"]

# =============================================================================
# Folder Creation
# =============================================================================
def create_folders():
    """Creates necessary folders"""
    os.makedirs(photo_folder, exist_ok=True)
    os.makedirs(settings_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)

# Create folders on program start
create_folders()