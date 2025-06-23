# -*- coding: utf-8 -*-
"""
Center Panel - Thumbnail list, engraving number input, and AI analysis button
"""

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import config
from .widgets import ClickableLabel


class CenterPanel:
    """Center panel containing thumbnail images and AI analysis controls"""
    
    def __init__(self, main_window):
        self.main_window = main_window
        self.image_label_list = []
        
    def create_center_panel(self):
        """Creates center panel"""
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(5, 5, 5, 5)
        
        # Scroll area
        thumbnail_scroll = QScrollArea()
        thumbnail_scroll.setWidgetResizable(True)
        thumbnail_content = QWidget()
        thumbnail_layout = QVBoxLayout(thumbnail_content)
        thumbnail_layout.setSpacing(1)
        
        # Image thumbnail labels (9 items)
        for i in range(9):
            label = ClickableLabel(index=i)
            label.setText(f"Photo{i+1}")
            label.setAlignment(Qt.AlignCenter)
            label.setFixedSize(155, 117)
            label.setStyleSheet("border: 1px solid gray; background-color: white;")
            thumbnail_layout.addWidget(label)
            self.image_label_list.append(label)
        
        thumbnail_scroll.setWidget(thumbnail_content)
        center_layout.addWidget(thumbnail_scroll)
        
        # Engraving number input
        engraving_layout = QHBoxLayout()
        engraving_layout.setSpacing(5)
        engraving_label = QLabel("Engraving No:")
        engraving_layout.addWidget(engraving_label)
        
        engraving_input = QLineEdit()
        engraving_input.setPlaceholderText("Enter engraving number")
        # Fix: Safe initialization - use empty string if engraving_number_input doesn't exist
        try:
            if hasattr(self.main_window, 'engraving_number_input'):
                engraving_input.setText(self.main_window.engraving_number_input.text())
            else:
                engraving_input.setText("")
        except:
            engraving_input.setText("")
            
        engraving_input.setFixedWidth(100)
        # Fix: Safe connection - add safety check in lambda
        engraving_input.textChanged.connect(
            lambda text: self.sync_engraving_input(text)
        )
        engraving_layout.addWidget(engraving_input)
        engraving_layout.addStretch(1)
        center_layout.addLayout(engraving_layout)
        
        # AI analysis button
        ai_analysis_button = QPushButton('AI Auto Analysis')
        ai_analysis_button.setFixedSize(200, 70)
        ai_analysis_button.clicked.connect(self.main_window.run_ai_analysis)
        ai_analysis_button.setStyleSheet(config.ai_analysis_button_style)
        center_layout.addWidget(ai_analysis_button, 0, Qt.AlignLeft)
        
        return center_panel
    
    def sync_engraving_input(self, text):
        """Safely sync engraving input with main window"""
        try:
            if hasattr(self.main_window, 'engraving_number_input'):
                self.main_window.engraving_number_input.setText(text)
        except:
            pass