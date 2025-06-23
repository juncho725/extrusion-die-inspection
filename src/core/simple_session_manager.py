# simple_session_manager.py
# Session Management - GUI Compatible Version

import os
import time
import json
from PyQt5.QtCore import QDate
from PyQt5.QtWidgets import QMessageBox, QInputDialog
import pandas as pd
import config

class SimpleSessionManager:
    """Simple session manager - GUI compatible"""
    
    def __init__(self, parent):
        self.parent = parent
        self.current_session_folder = None
        self.current_capture_folder = None
        self.current_results_folder = None
        self.settings = self.get_default_settings()
    
    def get_default_settings(self):
        """Get default settings"""
        return {
            "creation_date": QDate.currentDate(),
            "coating_date": QDate.currentDate(),
            "die_number": "",
            "engraving_number": "",
            "target_distances": config.default_target_distance.copy(),
            "tolerance_plus": config.default_tolerance_plus.copy(),
            "tolerance_minus": config.default_tolerance_minus.copy()
        }
    
    def start_new_session(self):
        """Start new session"""
        session_id = time.strftime("%Y%m%d_%H%M%S")
        self.current_session_folder = os.path.join(config.results_folder, f"session_{session_id}")
        self.current_capture_folder = os.path.join(self.current_session_folder, "captures")
        self.current_results_folder = os.path.join(self.current_session_folder, "results")
        
        os.makedirs(self.current_capture_folder, exist_ok=True)
        os.makedirs(self.current_results_folder, exist_ok=True)
        os.makedirs(os.path.join(self.current_results_folder, "visualizations"), exist_ok=True)
        
        self.settings = self.get_default_settings()
        return f"session_{session_id}"
    
    def save_settings(self, quietly=False):
        """Save settings"""
        die_number = self.settings["die_number"]
        
        if not die_number:
            die_number, ok = QInputDialog.getText(self.parent, "Die Number", "Enter die number:")
            if not ok or not die_number:
                return False
            self.settings["die_number"] = die_number
        
        # Save CSV
        os.makedirs(config.settings_folder, exist_ok=True)
        csv_path = os.path.join(config.settings_folder, f"{die_number}.csv")
        
        data = []
        for category in ["A", "B", "E", "F", "G"]:
            data.append({
                "category": category,
                "target": self.settings["target_distances"][category],
                "tolerance_plus": self.settings["tolerance_plus"][category],
                "tolerance_minus": self.settings["tolerance_minus"][category]
            })
        
        pd.DataFrame(data).to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # Save JSON
        if self.current_session_folder:
            json_path = os.path.join(self.current_session_folder, "settings.json")
            settings_copy = dict(self.settings)
            settings_copy["creation_date"] = self.settings["creation_date"].toString("yyyy-MM-dd")
            settings_copy["coating_date"] = self.settings["coating_date"].toString("yyyy-MM-dd")
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(settings_copy, f, indent=2)
        
        if not quietly:
            QMessageBox.information(self.parent, "Complete", f"Settings saved: {die_number}.csv")
        return True
    
    def load_die_settings(self, die_number):
        """Load settings by die number"""
        if not die_number:
            return False
        
        settings_path = os.path.join(config.settings_folder, f"{die_number}.csv")
        if not os.path.exists(settings_path):
            return False
        
        df = pd.read_csv(settings_path, encoding='utf-8-sig')
        self.settings["die_number"] = die_number
        
        for _, row in df.iterrows():
            category = row['category']
            if category in ['A', 'B', 'E', 'F', 'G']:
                if pd.notna(row.get('target')):
                    self.settings['target_distances'][category] = float(row['target'])
                if pd.notna(row.get('tolerance_plus')):
                    self.settings['tolerance_plus'][category] = float(row['tolerance_plus'])
                if pd.notna(row.get('tolerance_minus')):
                    self.settings['tolerance_minus'][category] = float(row['tolerance_minus'])
        
        return True
    
    def update_from_ui(self, main_window):
        """Update settings from UI"""
        self.settings["die_number"] = main_window.die_number_input.text()
        self.settings["engraving_number"] = main_window.engraving_number_input.text()
        self.settings["creation_date"] = main_window.creation_date_input.date()
        self.settings["coating_date"] = main_window.coating_date_input.date()
        
        for category in ["A", "B", "E", "F", "G"]:
            self.settings["target_distances"][category] = main_window.target_distance_inputs[category].value()
            self.settings["tolerance_plus"][category] = main_window.tolerance_plus_inputs[category].value()
            self.settings["tolerance_minus"][category] = main_window.tolerance_minus_inputs[category].value()
    
    def update_ui(self, main_window):
        """Update UI from settings"""
        main_window.die_number_input.setText(self.settings['die_number'])
        main_window.engraving_number_input.setText(self.settings['engraving_number'])
        main_window.creation_date_input.setDate(self.settings['creation_date'])
        main_window.coating_date_input.setDate(self.settings['coating_date'])
        
        for category in ["A", "B", "E", "F", "G"]:
            main_window.target_distance_inputs[category].setValue(self.settings["target_distances"][category])
            main_window.tolerance_plus_inputs[category].setValue(self.settings["tolerance_plus"][category])
            main_window.tolerance_minus_inputs[category].setValue(self.settings["tolerance_minus"][category])