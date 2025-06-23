# simple_image_manager.py
# Image Management - GUI Compatible Version

import os
import time
import pyautogui
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox
import config

class CaptureThread(QThread):
    """Simple capture thread"""
    finished = pyqtSignal(object)
    
    def __init__(self, region=None):
        super().__init__()
        self.region = region
    
    def run(self):
        if self.region:
            x, y, w, h = self.region
            screenshot = pyautogui.screenshot(region=(x, y, w, h))
        else:
            screenshot = pyautogui.screenshot()
        self.finished.emit(screenshot)

class SimpleImageManager:
    """Simple image manager - GUI compatible"""
    
    def __init__(self, parent):
        self.parent = parent
        self.images = [None] * 9
        self.current_preview_index = 0
        self.capture_count = 0
    
    def capture_image(self, save_folder):
        """Capture image"""
        region = (config.capture_x_coord, config.capture_y_coord, 
                 config.capture_width, config.capture_height)
        
        self.capture_thread = CaptureThread(region)
        self.capture_thread.finished.connect(lambda img: self.save_image(img, save_folder))
        self.capture_thread.start()
    
    def save_image(self, screenshot, save_folder):
        """Save image"""
        self.capture_count += 1
        slot = ((self.capture_count - 1) % 9)
        
        filename = f"capture_{slot + 1}.png"
        file_path = os.path.join(save_folder, filename)
        
        screenshot.save(file_path)
        self.images[slot] = file_path
        self.current_preview_index = slot
        
        self.update_single_thumbnail(slot, file_path)
        self.update_border()
        
        self.parent.statusBar().showMessage(f"Image {slot + 1} captured")
        
        if all(img is not None for img in self.images):
            QMessageBox.information(self.parent, "Complete", "9 images completed!")
    
    def delete_current_image(self):
        """Delete current image"""
        if self.images[self.current_preview_index] is None:
            QMessageBox.warning(self.parent, "Warning", "No image to delete.")
            return
        
        reply = QMessageBox.question(self.parent, "Delete", "Do you want to delete this image?")
        if reply == QMessageBox.Yes:
            os.remove(self.images[self.current_preview_index])
            self.images[self.current_preview_index] = None
            self.update_preview()
    
    def view_previous_image(self):
        """Move to previous image"""
        self.current_preview_index = (self.current_preview_index - 1) % 9
        self.update_border()
    
    def view_next_image(self):
        """Move to next image"""
        self.current_preview_index = (self.current_preview_index + 1) % 9
        self.update_border()
    
    def view_current_image(self):
        """View current image"""
        self.update_border()
    
    def thumbnail_click(self, index):
        """Thumbnail click"""
        self.current_preview_index = index
        self.update_border()
    
    def update_preview(self):
        """Update preview"""
        for i, image_path in enumerate(self.images):
            label = self.parent.screen_layout.image_label_list[i]
            
            if image_path and os.path.exists(image_path):
                pixmap = QPixmap(image_path)
                scaled = pixmap.scaled(label.width(), label.height(), 
                                     Qt.KeepAspectRatio, Qt.SmoothTransformation)
                label.setPixmap(scaled)
            else:
                label.clear()
                label.setText(f"Photo{i + 1}")
        
        self.update_border()
    
    def update_border(self):
        """Update border"""
        for i, label in enumerate(self.parent.screen_layout.image_label_list):
            if i == self.current_preview_index:
                label.setStyleSheet("border: 2px solid red; background-color: white;")
            else:
                label.setStyleSheet("border: 1px solid gray; background-color: white;")
    
    def is_complete(self):
        """Check if 9 images are complete"""
        return all(img is not None for img in self.images)
    
    def reset(self):
        """Reset image set"""
        self.images = [None] * 9
        self.current_preview_index = 0
        self.capture_count = 0
        
        for i, label in enumerate(self.parent.screen_layout.image_label_list):
            label.clear()
            label.setText(f"Photo{i + 1}")
            label.setStyleSheet("border: 1px solid gray; background-color: white;")
    
    def update_single_thumbnail(self, index, image_path):
        """Update single thumbnail"""
        if index < len(self.parent.screen_layout.image_label_list):
            label = self.parent.screen_layout.image_label_list[index]
            pixmap = QPixmap(image_path)
            scaled = pixmap.scaled(label.width(), label.height(), 
                                Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(scaled)