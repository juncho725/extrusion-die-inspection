# -*- coding: utf-8 -*-
"""
Right Panel - Result images and distance measurement table
"""

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *


class RightPanel:
    """Right panel containing result display and measurement table"""
    
    def __init__(self, main_window):
        self.main_window = main_window
        self.test_label_list = []
        
    def create_right_panel(self):
        """Creates right panel"""
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)
        
        # Result image scroll area
        result_scroll = QScrollArea()
        result_scroll.setWidgetResizable(True)
        result_scroll.setMaximumHeight(220)
        
        result_widget = QWidget()
        result_grid = QGridLayout(result_widget)
        result_grid.setSpacing(5)
        
        # Test result labels (9 items)
        for i in range(9):
            row = i // 3
            col = i % 3
            label = QLabel("")
            label.setAlignment(Qt.AlignCenter)
            label.setFixedSize(250, 170)
            label.setStyleSheet("border: 1px solid lightgray; background-color: black; color: white;")
            result_grid.addWidget(label, row, col)
            self.test_label_list.append(label)
        
        result_scroll.setWidget(result_widget)
        right_layout.addWidget(result_scroll)
        
        # Distance data table
        self.main_window.distance_table = QTableWidget()
        self.main_window.distance_table.setMinimumHeight(100)
        self.main_window.distance_table.horizontalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.main_window.distance_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.main_window.distance_table.customContextMenuRequested.connect(self.main_window.show_table_context_menu)
        right_layout.addWidget(self.main_window.distance_table)
        
        return right_panel