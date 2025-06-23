# simple_data_manager.py
# Data Management - GUI Compatible Version

import os
import pandas as pd
from PyQt5.QtWidgets import QTableWidgetItem
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QBrush, QColor

class SimpleDataManager:
    """Simple data manager - GUI compatible"""
    
    def __init__(self, parent):
        self.parent = parent
        self.current_column = 0
        self.engraving_numbers = {}
    
    def create_table(self, results_folder, settings):
        """Create table"""
        categories = ["A00", "B00", "E01", "F02"] + [f"G{i:02d}" for i in range(3, 16)] + ["F16", "E17"]
        
        data = []
        
        # Product name row
        engraving = settings.get("engraving_number", "")
        data.append({
            "Distance_Category": "Product Name",
            "Target_Distance": None,
            "Tolerance_Plus": None,
            "Tolerance_Minus": None,
            "Product_0": engraving if engraving else ""
        })
        
        # Each category row
        for cat in categories:
            base_cat = cat[0]
            data.append({
                "Distance_Category": cat,
                "Target_Distance": settings["target_distances"].get(base_cat, 0.25),
                "Tolerance_Plus": settings["tolerance_plus"].get(base_cat, 0.01),
                "Tolerance_Minus": settings["tolerance_minus"].get(base_cat, 0.01),
                "Product_0": None
            })
        
        # Create and save dataframe
        df = pd.DataFrame(data)
        csv_path = os.path.join(results_folder, "vertical_object_distances.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # Display in UI
        self.display_data_in_gui_table(df, settings)
        
        return df
    
    def display_data_in_gui_table(self, df, settings):
        """Display data in GUI table"""
        table = self.parent.distance_table
        
        # Basic setup
        table.clear()
        table.setColumnCount(14)
        
        # Data rows only (excluding Product Name)
        data_rows = df[df['Distance_Category'] != 'Product Name']
        table.setRowCount(2 + len(data_rows) * 2)
        
        # Header setup
        header_texts = ['No.', 'Item', 'Tolerance\n', 'Target\nDimension']
        for col, text in enumerate(header_texts):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignCenter)
            item.setBackground(QBrush(QColor(220, 220, 220)))
            item.setFont(QFont("Arial", 12, QFont.Bold))
            table.setItem(0, col, item)
            table.setSpan(0, col, 2, 1)
        
        # Engraving number header
        header_item = QTableWidgetItem('Engraving No.')
        header_item.setTextAlignment(Qt.AlignCenter)
        header_item.setBackground(QBrush(QColor(220, 220, 220)))
        header_item.setFont(QFont("Arial", 12, QFont.Bold))
        table.setItem(0, 4, header_item)
        table.setSpan(0, 4, 1, 10)
        
        # Engraving number values
        for i in range(10):
            product_column_name = f'Product_{i}'
            if product_column_name in df.columns:
                product_value = df.loc[df['Distance_Category'] == 'Product Name', product_column_name].values
                
                if len(product_value) > 0 and not pd.isna(product_value[0]):
                    item = QTableWidgetItem(str(product_value[0]))
                    item.setTextAlignment(Qt.AlignCenter)
                    if i == self.current_column:
                        item.setBackground(QBrush(QColor(245, 245, 245)))
                    else:
                        item.setBackground(QBrush(QColor(220, 220, 220)))
                    item.setFont(QFont("Arial", 12, QFont.Bold))
                    table.setItem(1, 4 + i, item)
        
        # Add data rows
        for i, (_, row) in enumerate(data_rows.iterrows()):
            category = row['Distance_Category']
            basic_row = i * 2 + 2
            basic_category = category[0] if category else ""
            
            # Number
            number_item = QTableWidgetItem(f"{i+1:02d}")
            number_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(basic_row, 0, number_item)
            table.setSpan(basic_row, 0, 2, 1)
            
            # Item
            item_cell = QTableWidgetItem(category)
            item_cell.setTextAlignment(Qt.AlignCenter)
            item_cell.setFont(QFont("Arial", 12, QFont.Bold))
            item_cell.setBackground(QBrush(QColor(240, 240, 240)))
            table.setItem(basic_row, 1, item_cell)
            table.setSpan(basic_row, 1, 2, 1)
            
            # Tolerance
            tolerance_plus_val = settings["tolerance_plus"].get(basic_category, 0.010)
            tolerance_minus_val = settings["tolerance_minus"].get(basic_category, 0.010)
            
            plus_item = QTableWidgetItem(f"+ {tolerance_plus_val:.3f}")
            plus_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            table.setItem(basic_row, 2, plus_item)
            
            minus_item = QTableWidgetItem(f"- {tolerance_minus_val:.3f}")
            minus_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            table.setItem(basic_row + 1, 2, minus_item)
            
            # Target value
            target_value = settings["target_distances"].get(basic_category, 0.24 if basic_category == 'G' else 0.25)
            target_item = QTableWidgetItem(f"{target_value:.3f}")
            target_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            table.setItem(basic_row, 3, target_item)
            table.setSpan(basic_row, 3, 2, 1)
            
            # Display product column measurement values
            for product_index in range(10):
                product_col_ui = 4 + product_index
                product_column_name = f"Product_{product_index}"
                
                if product_column_name in df.columns:
                    value = row.get(product_column_name)
                    if pd.notna(value) and value != "":
                        value_item = QTableWidgetItem(str(value))
                        value_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                        table.setItem(basic_row, product_col_ui, value_item)
                        table.setSpan(basic_row, product_col_ui, 2, 1)
        
        # Apply styling
        table.setShowGrid(True)
        table.setStyleSheet("QTableWidget { gridline-color: black; }")
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setVisible(False)
        
        # Row height and column width
        for row in range(table.rowCount()):
            table.setRowHeight(row, 30)
        
        column_widths = {0: 40, 1: 50, 2: 60, 3: 70}
        for col, width in column_widths.items():
            table.setColumnWidth(col, width)
        
        for col in range(4, table.columnCount()):
            table.setColumnWidth(col, 60)
        
        table.setColumnHidden(0, True)
        
        self.highlight_current_column()
    
    def update_product_column(self, results_folder, column_index, engraving_number):
        """Update product column"""
        csv_path = os.path.join(results_folder, "vertical_object_distances.csv")
        if not os.path.exists(csv_path):
            return False
        
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        
        # Update product name
        product_col = f'Product_{column_index}'
        if product_col not in df.columns:
            df[product_col] = ""
        
        if not engraving_number:
            engraving_number = "-"
        elif engraving_number.isdigit():
            engraving_number = f" {engraving_number}"
        
        df.loc[df['Distance_Category'] == 'Product Name', product_col] = engraving_number
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        self.engraving_numbers[column_index] = engraving_number
        return True
    
    def move_to_next_product(self):
        """Move to next product"""
        if self.current_column < 9:
            self.current_column += 1
            self.highlight_current_column()
            return True
        return False
    
    def move_to_previous_product(self):
        """Move to previous product"""
        if self.current_column > 0:
            self.current_column -= 1
            self.highlight_current_column()
            return True
        return False
    
    def highlight_current_column(self):
        """Highlight current column"""
        table = self.parent.distance_table
        
        if table.columnCount() < 5:
            return
        
        # Set all product columns to gray
        for i in range(10):
            col = 4 + i
            item = table.item(1, col)
            
            if not item:
                item = QTableWidgetItem("")
                item.setTextAlignment(Qt.AlignCenter)
                item.setFont(QFont("Arial", 12, QFont.Bold))
                table.setItem(1, col, item)
            
            if i != self.current_column:
                item.setBackground(QBrush(QColor(220, 220, 220)))
        
        # Set current column to light color
        current_col = 4 + self.current_column
        current_item = table.item(1, current_col)
        
        if not current_item:
            current_item = QTableWidgetItem("")
            current_item.setTextAlignment(Qt.AlignCenter)
            current_item.setFont(QFont("Arial", 12, QFont.Bold))
            table.setItem(1, current_col, current_item)
        
        current_item.setBackground(QBrush(QColor(245, 245, 245)))
        table.update()
    
    def add_analysis_results(self, results_folder, ai_results):
        """Add AI analysis results"""
        csv_path = os.path.join(results_folder, "vertical_object_distances.csv")
        if not os.path.exists(csv_path):
            return False
        
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        
        # Add results
        product_col = f'Product_{self.current_column}'
        if product_col not in df.columns:
            df[product_col] = ""
        
        for _, result_row in ai_results.iterrows():
            category = result_row['Category']
            value = result_row['Scaled_Distance']
            
            df.loc[df['Distance_Category'] == category, product_col] = f"{value:.3f}"
        
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        return True
    
    def get_current_engraving(self):
        """Get current column's engraving number"""
        return self.engraving_numbers.get(self.current_column, "")