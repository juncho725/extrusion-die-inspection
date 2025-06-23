# simple_ai_manager.py
# AI Analysis Management - GUI Compatible Version

import os
import sys
import subprocess
import importlib.util
import shutil
from PyQt5.QtWidgets import QMessageBox, QPushButton, QApplication, QInputDialog, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import config

class SimpleAIManager:
    """Simple AI manager - GUI compatible"""
    
    def __init__(self, parent):
        self.parent = parent
        self.is_running = False
    
    def run_ai_analysis(self):
        """AI analysis execution - for GUI call"""
        if not self.image_manager.is_complete():
            QMessageBox.warning(self, "Warning", "9 images required.")
            return
        
        # Check and create table
        table_path = os.path.join(self.session_manager.current_results_folder, "vertical_object_distances.csv")
        if not os.path.exists(table_path):
            reply = QMessageBox.question(self, "Create Table", "Do you want to create analysis table?")
            if reply == QMessageBox.Yes:
                self.create_table()
            else:
                return
        
        # Save current engraving number
        current_engraving = self.engraving_number_input.text().strip()
        self.data_manager.update_product_column(
            self.session_manager.current_results_folder,
            self.data_manager.current_column,
            current_engraving
        )
        
        self.session_manager.update_from_ui(self)
        
        results = self.ai_manager.run_ai_analysis(
            self.session_manager.current_capture_folder,
            self.session_manager.current_results_folder
        )
        
        if results is not None:
            self.data_manager.add_analysis_results(self.session_manager.current_results_folder, results)
            df = pd.read_csv(table_path, encoding='utf-8-sig')
            self.data_manager.display_data_in_gui_table(df, self.session_manager.settings)    
            self.display_test_results()
            self.ai_manager.update_preview_images(self.session_manager.current_capture_folder)
            self.statusBar().showMessage("AI analysis complete!")

    def run_ai_analysis(self, capture_folder, results_folder):
        """Execute AI analysis - for GUI call"""
        # Set analysis state
        self.set_analysis_state(True)
        
        try:
            # Execute image rotation
            self.execute_rotation(capture_folder)
            
            # Execute AI analysis
            results = self.execute_ai_analysis(capture_folder, results_folder)
            
            return results
            
        except Exception as e:
            QMessageBox.critical(self.parent, "Error", f"AI analysis failed: {str(e)}")
            return None
        finally:
            # Reset analysis state
            self.set_analysis_state(False)

    def execute_rotation(self, capture_folder):
        """Execute image rotation"""
        if not os.path.exists(config.rotation_code_path):
            return True
        
        os.environ["INPUT_DIR"] = capture_folder
        
        try:
            if getattr(sys, 'frozen', False):
                spec = importlib.util.spec_from_file_location("rotation", config.rotation_code_path)
                rotation_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(rotation_module)
                rotation_module.rotate_capture_set(capture_folder)
            else:
                subprocess.run([sys.executable, config.rotation_code_path], 
                             capture_output=True, text=True)
            return True
        except:
            return True  # Continue even if rotation fails
    
    def execute_ai_analysis(self, capture_folder, results_folder):
        """Execute AI analysis"""
        if not os.path.exists(config.test_code_path):
            QMessageBox.warning(self.parent, "Warning", "AI test code not found.")
            return None
        
        spec = importlib.util.spec_from_file_location("test_module", config.test_code_path)
        test_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_module)
        
        model = test_module.load_model(config.model_file_path)
        results = test_module.test_and_visualize(
            model, capture_folder, results_folder, threshold=0.3
        )
        
        return results
    
    def display_test_results(self, results_folder):
        """Display test results"""
        viz_folder = os.path.join(results_folder, "visualizations")
        if not os.path.exists(viz_folder):
            return
        
        # Find result images
        result_files = []
        for i in range(1, 10):
            file_path = os.path.join(viz_folder, f"capture_{i}.png")
            if os.path.exists(file_path):
                result_files.append((i, file_path))
        
        # If none found, get all image files
        if not result_files:
            all_files = [f for f in os.listdir(viz_folder) if f.endswith('.png')]
            result_files = [(i+1, os.path.join(viz_folder, f)) for i, f in enumerate(all_files[:9])]
        
        # Display in UI
        for i, (number, file_path) in enumerate(result_files):
            if i < len(self.parent.screen_layout.test_label_list):
                pixmap = QPixmap(file_path)
                label = self.parent.screen_layout.test_label_list[i]
                scaled = pixmap.scaled(label.width(), label.height(), 
                                     Qt.KeepAspectRatio, Qt.SmoothTransformation)
                label.setPixmap(scaled)
                label.setToolTip(f"Capture {number}")
    
    def export_images(self, results_folder):
        """Export result images"""
        viz_folder = os.path.join(results_folder, "visualizations")
        if not os.path.exists(viz_folder):
            QMessageBox.warning(self.parent, "Warning", "No results to export.")
            return
        
        # Input folder name
        folder_name, ok = QInputDialog.getText(
            self.parent, "Folder Name", "Save folder name:", text="AI_Analysis_Results"
        )
        if not ok or not folder_name:
            return
        
        # Select save location
        save_location = QFileDialog.getExistingDirectory(
            self.parent, "Select Save Location", os.path.expanduser("~/Desktop")
        )
        if not save_location:
            return
        
        # Copy images
        target_path = os.path.join(save_location, folder_name)
        shutil.copytree(viz_folder, target_path, dirs_exist_ok=True)
        
        # Completion message and open folder
        QMessageBox.information(self.parent, "Complete", f"Results saved!\n{target_path}")
        os.startfile(target_path)
    
    def set_analysis_state(self, is_running):
        """Set analysis state"""
        self.is_running = is_running
        
        # Find AI button and change state
        for button in self.parent.findChildren(QPushButton):
            if "AI Auto Analysis" in button.text() or "analyzing" in button.text().lower():
                if is_running:
                    button.setText('Analyzing...')
                    button.setStyleSheet("QPushButton { background-color: #888888; color: white; font-weight: 900; font-size: 25px; }")
                    button.setEnabled(False)
                else:
                    button.setText('AI Auto Analysis')
                    button.setStyleSheet(config.ai_analysis_button_style)
                    button.setEnabled(True)
                break
        
        QApplication.processEvents()
    
    def update_preview_images(self, capture_folder):
        """Update preview images"""
        for i in range(9):
            image_path = os.path.join(capture_folder, f"capture_{i+1}.png")
            if os.path.exists(image_path) and i < len(self.parent.screen_layout.image_label_list):
                pixmap = QPixmap(image_path)
                label = self.parent.screen_layout.image_label_list[i]
                scaled = pixmap.scaled(label.width(), label.height(), 
                                     Qt.KeepAspectRatio, Qt.SmoothTransformation)
                label.setPixmap(scaled)