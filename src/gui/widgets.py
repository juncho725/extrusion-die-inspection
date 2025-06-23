# -*- coding: utf-8 -*-
"""
Custom Widgets - Defines custom UI components
"""

from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt


class ClickableLabel(QLabel):
    """Clickable label widget for thumbnail images"""
    
    def __init__(self, parent=None, index=0):
        super().__init__(parent)
        self.index = index
        
    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if event.button() == Qt.LeftButton:
            # Call parent object's thumbnail click method
            main_window = self.parent().parentWidget().parentWidget()
            if hasattr(main_window, 'thumbnail_click'):
                main_window.thumbnail_click(self.index)