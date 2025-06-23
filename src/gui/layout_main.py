# -*- coding: utf-8 -*-
"""
Main Layout Manager - Assembles all panels into the main screen
"""

import os
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import config
from .panels_left import LeftPanel
from .panels_center import CenterPanel
from .panels_right import RightPanel


class ScreenLayout:
    """Class responsible for UI composition of main window"""
    
    def __init__(self, main_window):
        self.main_window = main_window
        
        # Initialize panel managers
        self.left_panel = LeftPanel(main_window)
        self.center_panel = CenterPanel(main_window)
        self.right_panel = RightPanel(main_window)
        
        # Initialize label lists for compatibility
        self.image_label_list = []
        self.test_label_list = []
        
    def ui_initialization(self):
        """Initializes entire UI"""
        self.basic_window_settings()
        self.create_main_layout()
        
        # Set up label lists after panels are created
        self.image_label_list = self.center_panel.image_label_list
        self.test_label_list = self.right_panel.test_label_list
        
    def basic_window_settings(self):
        """Basic window settings"""
        self.main_window.setWindowTitle('Extrusion Die Automatic Inspection Program')
        window_width = 1300
        window_height = config.capture_y_coord + config.capture_height + 300
        self.main_window.setGeometry(50, 50, window_width, window_height)
        
        # Basic font settings
        app_font = QFont()
        app_font.setPointSize(12)
        app_font.setBold(True)
        QApplication.instance().setFont(app_font)
        
        # Apply overall style
        self.main_window.setStyleSheet(config.button_style)
        
        # Frameless window settings
        self.main_window.setWindowFlags(Qt.FramelessWindowHint)
        self.main_window.setAttribute(Qt.WA_TranslucentBackground, True)
        
    def create_main_layout(self):
        """Creates main layout"""
        main_widget = QWidget()
        self.main_window.setCentralWidget(main_widget)
        
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create top menu bar
        top_menu = self.create_top_menu_bar()
        main_layout.addWidget(top_menu)
        
        # Create content area
        content_area = self.create_content_area()
        main_layout.addWidget(content_area)
        
        # Add status bar
        self.main_window.statusBar().showMessage("Program started")
        
    def create_top_menu_bar(self):
        """Creates top menu bar"""
        top_menu_widget = QWidget()
        top_menu_widget.setStyleSheet("background-color: #f0f0f0;")
        top_menu_widget.setFixedHeight(60)
        
        top_layout = QHBoxLayout(top_menu_widget)
        top_layout.setContentsMargins(10, 15, 10, 5)
        top_layout.addSpacing(20)
        
        # Add company logo
        if os.path.exists(config.company_logo_path):
            image_label = QLabel()
            pixmap = QPixmap(config.company_logo_path)
            scaled_pixmap = pixmap.scaled(300, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            image_label.setPixmap(scaled_pixmap)
            image_label.setAlignment(Qt.AlignCenter)
            top_layout.addWidget(image_label)
        
        top_layout.addSpacing(400)
        
        # Title text
        title_label = QLabel("Extrusion Die Automatic Inspection Device")
        title_label.setFont(QFont("Arial", 23, QFont.Bold))
        title_label.setAlignment(Qt.AlignVCenter)
        title_label.setStyleSheet("color: #000000;")
        top_layout.addWidget(title_label)
        
        top_layout.addStretch(1)
        
        # Maximize/restore button
        maximize_button = QPushButton('[ ]')
        maximize_button.setFont(QFont("Arial", 24, QFont.Bold))
        maximize_button.setFixedSize(90, 45)
        maximize_button.setStyleSheet("QPushButton { background-color: #3498db; color: white; border-radius: 12px; }")
        maximize_button.clicked.connect(self.main_window.toggle_maximize)
        top_layout.addWidget(maximize_button)
        
        # Close button
        close_button = QPushButton('X')
        close_button.setFixedSize(90, 45)
        close_button.setStyleSheet("QPushButton { background-color: #e74c3c; color: white; border-radius: 12px; }")
        close_button.clicked.connect(self.main_window.close)
        top_layout.addWidget(close_button)
        
        return top_menu_widget
        
    def create_content_area(self):
        """Creates content area"""
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        
        # Left panel (capture area + information input)
        left_panel = self.left_panel.create_left_panel()
        content_layout.addWidget(left_panel)
        
        # Center panel (image list)
        center_panel = self.center_panel.create_center_panel()
        content_layout.addWidget(center_panel)
        
        # Right panel (result display)
        right_panel = self.right_panel.create_right_panel()
        content_layout.addWidget(right_panel)
        
        # Set ratios
        content_layout.setStretch(0, 12)  # Left panel
        content_layout.setStretch(1, 1)   # Center panel
        content_layout.setStretch(2, 12)  # Right panel
        
        return content_widget