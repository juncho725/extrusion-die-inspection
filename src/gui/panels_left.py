# -*- coding: utf-8 -*-
"""
Left Panel - Session management, image capture, and die information input UI
"""

import os
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import config


class LeftPanel:
    """Left panel containing capture area and information input"""
    
    def __init__(self, main_window):
        self.main_window = main_window
        
    def create_left_panel(self):
        """Creates left panel"""
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        # Top menu buttons
        top_button_layout = self.create_top_button_layout()
        left_layout.addLayout(top_button_layout)
        left_layout.addSpacing(5)
        
        # Capture area and navigation buttons
        capture_layout = self.create_capture_area()
        left_layout.addLayout(capture_layout)
        
        # Reduce vertical spacing (to increase Die Information size)
        left_layout.addSpacing(0)
        
        # Horizontal container: reduce horizontal movement
        info_container = QHBoxLayout()
        info_container.setContentsMargins(0, 0, 0, 0)
        info_container.addSpacing(15)
        
        # Information input area
        info_frame = self.create_info_input_area()
        info_container.addWidget(info_frame)
        info_container.addStretch(1)
        
        left_layout.addLayout(info_container)
        
        # Spacing with bottom buttons
        left_layout.addSpacing(10)
        
        # Bottom buttons
        bottom_button_layout = self.create_bottom_button_layout()
        left_layout.addLayout(bottom_button_layout)
        
        return left_panel
    
    def create_top_button_layout(self):
        """Creates top menu buttons"""
        top_button_layout = QHBoxLayout()
        top_button_layout.setSpacing(10)
        top_button_layout.addSpacing(30)
        
        # New button
        new_button = QPushButton("New")
        new_button.setStyleSheet(config.menu_button_style)
        new_button.clicked.connect(self.main_window.start_new_session)
        new_button.setFixedSize(100, 70)
        top_button_layout.addWidget(new_button)
        
        # Open button
        open_button = QPushButton("Open")
        open_button.setStyleSheet(config.menu_button_style)
        open_button.clicked.connect(self.main_window.load_session)
        open_button.setFixedSize(100, 70)
        top_button_layout.addWidget(open_button)
        
        # Save button
        save_button = QPushButton("Save")
        save_button.setStyleSheet(config.menu_button_style)
        save_button.clicked.connect(self.main_window.save_session_settings)
        save_button.setFixedSize(100, 70)
        top_button_layout.addWidget(save_button)
        
        # Separator
        top_button_layout.addWidget(QLabel("|"))
        
        # Export photos button
        export_photos_button = QPushButton("Export Photos")
        export_photos_button.setStyleSheet(config.menu_button_style)
        export_photos_button.setFixedSize(200, 70)
        export_photos_button.clicked.connect(self.main_window.export_images)
        top_button_layout.addWidget(export_photos_button)
        
        # Export excel button
        export_excel_button = QPushButton('Export Excel')
        export_excel_button.setStyleSheet(config.menu_button_style)
        export_excel_button.clicked.connect(self.main_window.generate_excel_report)
        export_excel_button.setFixedSize(240, 70)
        top_button_layout.addWidget(export_excel_button)
        
        top_button_layout.addStretch(1)
        
        return top_button_layout
        
    def create_capture_area(self):
        """Creates capture area and navigation buttons"""
        capture_layout = QHBoxLayout()
        capture_layout.setSpacing(20)
        
        # Capture area (transparent space)
        capture_space = QWidget()
        capture_space.setFixedSize(config.capture_width, config.capture_height)
        capture_space.setStyleSheet("background-color: transparent; border: 1px solid transparent;")
        capture_layout.addWidget(capture_space)
        
        # Navigation buttons
        navigation_button_layout = self.create_navigation_buttons()
        capture_layout.addLayout(navigation_button_layout)
        
        return capture_layout
        
    def create_navigation_buttons(self):
        """Creates navigation buttons"""
        navigation_layout = QVBoxLayout()
        navigation_layout.setSpacing(10)
        navigation_layout.setContentsMargins(0, 0, 0, 0)
        navigation_layout.addSpacing(-8)
        
        # Previous photo button
        prev_button = QPushButton('Previous Photo')
        prev_button.setFixedSize(155, 110)
        prev_button.setStyleSheet("QPushButton { background-color: #f0f0f0; color: black; font-weight: 900; font-size: 25px; font-family: 'Arial Black', 'Arial Bold', Arial; }")
        prev_button.clicked.connect(self.main_window.view_previous_image)
        navigation_layout.addWidget(prev_button)
        
        # Next photo button
        next_button = QPushButton('Next Photo')
        next_button.setFixedSize(155, 110)
        next_button.setStyleSheet("QPushButton { background-color: #f0f0f0; color: black; font-weight: 900; font-size: 25px; font-family: 'Arial Black', 'Arial Bold', Arial; }")
        next_button.clicked.connect(self.main_window.view_next_image)
        navigation_layout.addWidget(next_button)
        
        # Delete button
        delete_button = QPushButton('Delete')
        delete_button.setFixedSize(155, 80)
        delete_button.setStyleSheet("QPushButton { background-color: #f0f0f0; color: black; font-weight: 900; font-size: 25px; font-family: 'Arial Black', 'Arial Bold', Arial; }")
        delete_button.clicked.connect(self.main_window.delete_current_image)
        navigation_layout.addWidget(delete_button)
        
        # Capture button
        capture_button = QPushButton('Take Photo')
        capture_button.setFixedSize(155, 160)
        capture_button.setStyleSheet(config.capture_button_style)
        capture_button.clicked.connect(self.main_window.capture_image)
        navigation_layout.addWidget(capture_button)
        
        return navigation_layout
        
    def create_info_input_area(self):
        """Creates information input area"""
        info_frame = QFrame()
        info_frame.setFrameShape(QFrame.Box)
        info_frame.setLineWidth(2)
        info_frame.setStyleSheet("border: 2px solid #888888; border-radius: 5px; background-color: transparent;")
        
        info_layout = QHBoxLayout(info_frame)
        info_layout.setContentsMargins(5, 3, 5, 3)
        
        # Left info (basic information)
        left_info_widget = self.create_left_info_widget()
        info_layout.addWidget(left_info_widget)
        
        # Right info (distance settings)
        right_info_widget = self.create_right_info_widget()
        info_layout.addWidget(right_info_widget)
        
        return info_frame
        
    def create_left_info_widget(self):
        """Creates left info widget"""
        left_info_widget = QWidget()
        left_info_layout = QVBoxLayout(left_info_widget)
        left_info_layout.setContentsMargins(5, 3, 5, 3)
        left_info_layout.addSpacing(3)
        
        # Title - significantly increase size
        title_label = QLabel("Die Info")
        title_label.setFont(QFont("Arial", 30, QFont.Bold))
        left_info_layout.addWidget(title_label)
        left_info_layout.addSpacing(3)
        
        # Input fields
        form_layout = QFormLayout()
        form_layout.setVerticalSpacing(6)
        form_layout.setHorizontalSpacing(3)
        
        # Creation date
        self.main_window.creation_date_input = QDateEdit()
        self.main_window.creation_date_input.setCalendarPopup(True)
        self.main_window.creation_date_input.setDate(QDate.currentDate())
        self.main_window.creation_date_input.setFixedWidth(130)
        self.main_window.creation_date_input.setFixedHeight(28)
        self.main_window.creation_date_input.setFont(QFont("Arial", 12))
        form_layout.addRow("Creation Date:", self.main_window.creation_date_input)
        
        # Coating date
        self.main_window.coating_date_input = QDateEdit()
        self.main_window.coating_date_input.setCalendarPopup(True)
        self.main_window.coating_date_input.setDate(QDate.currentDate())
        self.main_window.coating_date_input.setFixedWidth(130)
        self.main_window.coating_date_input.setFixedHeight(28)
        self.main_window.coating_date_input.setFont(QFont("Arial", 12))
        form_layout.addRow("Coating Date:", self.main_window.coating_date_input)
        
        # Die number
        self.main_window.die_number_input = QLineEdit()
        self.main_window.die_number_input.setPlaceholderText("e.g.: D1234")
        self.main_window.die_number_input.setFixedWidth(130)
        self.main_window.die_number_input.setFixedHeight(28)
        self.main_window.die_number_input.setFont(QFont("Arial", 12))
        self.main_window.die_number_input.editingFinished.connect(self.main_window.load_die_settings)
        form_layout.addRow("Die No.:", self.main_window.die_number_input)
        
        # Engraving number
        self.main_window.engraving_number_input = QLineEdit()
        self.main_window.engraving_number_input.setPlaceholderText("e.g.: E5678")
        self.main_window.engraving_number_input.setFixedWidth(130)
        self.main_window.engraving_number_input.setFixedHeight(28)
        self.main_window.engraving_number_input.setFont(QFont("Arial", 12))
        form_layout.addRow("Engraving No.:", self.main_window.engraving_number_input)
        
        left_info_layout.addLayout(form_layout)
        left_info_layout.addStretch(1)
        
        return left_info_widget
    
    def create_right_info_widget(self):
        """Creates right info widget"""
        right_info_widget = QWidget()
        right_info_layout = QVBoxLayout(right_info_widget)
        right_info_layout.setContentsMargins(5, 3, 5, 3)
        
        # Add drawing image - significantly increase size
        if os.path.exists(config.drawing_image_path):
            image_label = QLabel()
            pixmap = QPixmap(config.drawing_image_path)
            scaled_pixmap = pixmap.scaled(430, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            image_label.setPixmap(scaled_pixmap)
            image_label.setAlignment(Qt.AlignCenter)
            right_info_layout.addWidget(image_label)
        
        right_info_layout.addSpacing(5)
        
        # Distance setting grid
        distance_layout = self.create_distance_setting_grid()
        right_info_layout.addLayout(distance_layout)
        right_info_layout.addStretch(1)
        
        return right_info_widget
        
    def create_distance_setting_grid(self):
        """Creates distance setting grid"""
        distance_layout = QGridLayout()
        distance_layout.setSpacing(3)
        
        # Add headers
        item_label = QLabel("Item")
        item_label.setAlignment(Qt.AlignLeft)
        item_label.setFont(QFont("Arial", 10))
        distance_layout.addWidget(item_label, 0, 0)
        
        target_dimension_label = QLabel("Target")
        target_dimension_label.setAlignment(Qt.AlignRight)
        target_dimension_label.setFont(QFont("Arial", 10))
        distance_layout.addWidget(target_dimension_label, 0, 1)
        
        plus_tolerance_label = QLabel("Tol(+)")
        plus_tolerance_label.setAlignment(Qt.AlignRight)
        plus_tolerance_label.setFont(QFont("Arial", 10))
        distance_layout.addWidget(plus_tolerance_label, 0, 2)
        
        minus_tolerance_label = QLabel("Tol(-)")
        minus_tolerance_label.setAlignment(Qt.AlignRight)
        minus_tolerance_label.setFont(QFont("Arial", 10))
        distance_layout.addWidget(minus_tolerance_label, 0, 3)
        
        # Create input fields
        self.main_window.target_distance_inputs = {}
        self.main_window.tolerance_plus_inputs = {}
        self.main_window.tolerance_minus_inputs = {}
        
        for i, category in enumerate(["A", "B", "E", "F", "G"]):
            # Category label
            category_label = QLabel(category)
            category_label.setFont(QFont("Arial", 10, QFont.Bold))
            distance_layout.addWidget(category_label, i+1, 0)
            
            # Target distance input
            target_distance_input = QDoubleSpinBox()
            target_distance_input.setDecimals(3)
            target_distance_input.setRange(0.0, 20.0)
            target_distance_input.setSingleStep(0.001)
            target_distance_input.setFixedWidth(110)
            target_distance_input.setFixedHeight(20)
            target_distance_input.setFont(QFont("Arial", 9))
            target_distance_input.setAlignment(Qt.AlignRight)
            target_distance_input.setValue(config.default_target_distance[category])
            self.main_window.target_distance_inputs[category] = target_distance_input
            distance_layout.addWidget(target_distance_input, i+1, 1)
            
            # Plus tolerance input
            tolerance_plus_input = QDoubleSpinBox()
            tolerance_plus_input.setDecimals(3)
            tolerance_plus_input.setRange(0.001, 1.0)
            tolerance_plus_input.setSingleStep(0.001)
            tolerance_plus_input.setFixedWidth(110)
            tolerance_plus_input.setFixedHeight(20)
            tolerance_plus_input.setFont(QFont("Arial", 9))
            tolerance_plus_input.setAlignment(Qt.AlignRight)
            tolerance_plus_input.setValue(config.default_tolerance_plus[category])
            self.main_window.tolerance_plus_inputs[category] = tolerance_plus_input
            distance_layout.addWidget(tolerance_plus_input, i+1, 2)
            
            # Minus tolerance input
            tolerance_minus_input = QDoubleSpinBox()
            tolerance_minus_input.setDecimals(3)
            tolerance_minus_input.setRange(0.001, 1.0)
            tolerance_minus_input.setSingleStep(0.001)
            tolerance_minus_input.setFixedWidth(110)
            tolerance_minus_input.setFixedHeight(20)
            tolerance_minus_input.setFont(QFont("Arial", 9))
            tolerance_minus_input.setAlignment(Qt.AlignRight)
            tolerance_minus_input.setValue(config.default_tolerance_minus[category])
            self.main_window.tolerance_minus_inputs[category] = tolerance_minus_input
            distance_layout.addWidget(tolerance_minus_input, i+1, 3)
            
        return distance_layout
        
    def create_bottom_button_layout(self):
        """Creates bottom button layout"""
        bottom_button_layout = QHBoxLayout()
        bottom_button_layout.setSpacing(2)
        bottom_button_layout.setContentsMargins(25, 0, 0, 0)
        
        # Create table button
        create_table_button = QPushButton('Create Table')
        create_table_button.setFixedHeight(50)
        create_table_button.setFixedWidth(263)
        create_table_button.clicked.connect(self.main_window.create_table)
        create_table_button.setStyleSheet(config.product_nav_button_style)
        bottom_button_layout.addWidget(create_table_button)
        
        # Previous product button
        prev_product_button = QPushButton('Previous Product')
        prev_product_button.setFixedHeight(50)
        prev_product_button.setFixedWidth(263)
        prev_product_button.clicked.connect(self.main_window.move_to_previous_product)
        prev_product_button.setStyleSheet(config.product_nav_button_style)
        bottom_button_layout.addWidget(prev_product_button)
        
        # Next product button
        next_product_button = QPushButton('Next Product')
        next_product_button.setFixedHeight(50)
        next_product_button.setFixedWidth(263)
        next_product_button.clicked.connect(self.main_window.move_to_next_product)
        next_product_button.setStyleSheet(config.product_nav_button_style)
        bottom_button_layout.addWidget(next_product_button)
        
        return bottom_button_layout