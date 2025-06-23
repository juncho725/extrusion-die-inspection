#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# 프로젝트 루트에서 src 폴더를 Python path에 추가
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 이제 src 폴더의 모듈들을 import 가능
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import config
from gui import ScreenLayout
import pandas as pd 

# core 폴더의 모듈들을 절대 import로
from core.simple_session_manager import SimpleSessionManager
from core.simple_image_manager import SimpleImageManager
from core.simple_data_manager import SimpleDataManager
from core.simple_ai_manager import SimpleAIManager

class SimpleMainProgram(QMainWindow):
    """간단한 메인 프로그램 - GUI 완전 호환"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.init_managers()
        self.start_new_session()
        
    def init_ui(self):
        """UI 초기화"""
        self.screen_layout = ScreenLayout(self)
        self.screen_layout.ui_initialization()
        
        # GUI가 접근하는 속성들 설정
        self.distance_table = getattr(self, 'distance_table', None)

        self.showMaximized()
        
    def init_managers(self):
        """매니저들 초기화"""
        self.session_manager = SimpleSessionManager(self)
        self.image_manager = SimpleImageManager(self)
        self.data_manager = SimpleDataManager(self)
        self.ai_manager = SimpleAIManager(self)
    
    # =========================
    # GUI 호환용 함수들 (정확한 함수명)
    # =========================
    
    def start_new_session(self):
        """새 세션 시작"""
        session_name = self.session_manager.start_new_session()
        self.image_manager.reset()
        self.data_manager.current_column = 0
        self.data_manager.engraving_numbers = {}
        self.session_manager.update_ui(self)
        self.statusBar().showMessage(f"새 세션 시작: {session_name}")
    
    def save_session_settings(self, quietly=False):
        """설정 저장 - GUI 호출용"""
        self.session_manager.update_from_ui(self)
        if self.session_manager.save_settings(quietly):
            if not quietly:
                self.statusBar().showMessage("설정 저장 완료")
    
    def load_session(self):
        """세션 불러오기 - GUI 호출용"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "설정 파일 선택", config.settings_folder, "CSV Files (*.csv)"
        )
        
        if file_path:
            self.start_new_session()
            die_number = os.path.splitext(os.path.basename(file_path))[0]
            if self.session_manager.load_die_settings(die_number):
                self.session_manager.update_ui(self)
                self.statusBar().showMessage(f"설정 로드 완료: {die_number}")
    
    def load_die_settings(self):
        """금형 설정 로드 - GUI 호출용"""
        die_number = self.die_number_input.text().strip()
        if die_number and self.session_manager.load_die_settings(die_number):
            self.session_manager.update_ui(self)
            self.statusBar().showMessage(f"설정 자동 로드: {die_number}")
    
    def capture_image(self):
        """이미지 캡처 - GUI 호출용"""
        self.image_manager.capture_image(self.session_manager.current_capture_folder)
        self.statusBar().showMessage("이미지 캡처 중...")
    
    def delete_current_image(self):
        """현재 이미지 삭제 - GUI 호출용"""
        self.image_manager.delete_current_image()
    
    def view_previous_image(self):
        """이전 이미지 보기 - GUI 호출용"""
        self.image_manager.view_previous_image()
        self.statusBar().showMessage(f"이미지 {self.image_manager.current_preview_index + 1} 선택")
    
    def view_next_image(self):
        """다음 이미지 보기 - GUI 호출용"""
        self.image_manager.view_next_image()
        self.statusBar().showMessage(f"이미지 {self.image_manager.current_preview_index + 1} 선택")
    
    def view_current_image(self):
        """현재 이미지 보기 - GUI 호출용"""
        self.image_manager.view_current_image()
        self.statusBar().showMessage(f"이미지 {self.image_manager.current_preview_index + 1} 선택")
    
    def thumbnail_click(self, index):
        """썸네일 클릭 - GUI 호출용"""
        self.image_manager.thumbnail_click(index)
        self.statusBar().showMessage(f"이미지 {index + 1} 선택")
    
    def on_thumbnail_clicked(self, index):
        """기존 GUI 호환용 - 썸네일 클릭 처리"""
        self.thumbnail_click(index)
    def update_thumbnail_border(self):
        """썸네일 테두리 업데이트 - GUI 호출용"""
        self.image_manager.update_thumbnail_border()
    
    def create_table(self):
        """테이블 생성 - GUI 호출용"""
        self.session_manager.update_from_ui(self)
        self.data_manager.create_table(
            self.session_manager.current_results_folder,
            self.session_manager.settings
        )
        self.statusBar().showMessage("분석 테이블 생성 완료")
    
    def move_to_next_product(self):
        """다음 제품으로 이동 - GUI 호출용"""
        current_engraving = self.engraving_number_input.text().strip()
        self.data_manager.update_product_column(
            self.session_manager.current_results_folder,
            self.data_manager.current_column,
            current_engraving
        )
        
        if self.data_manager.move_to_next_product():
            next_engraving = self.data_manager.get_current_engraving()
            self.engraving_number_input.setText(next_engraving)
            self.statusBar().showMessage(f"제품 컬럼 {self.data_manager.current_column + 1}")
        else:
            QMessageBox.warning(self, "경고", "마지막 제품입니다.")
    
    def move_to_previous_product(self):
        """이전 제품으로 이동 - GUI 호출용"""
        current_engraving = self.engraving_number_input.text().strip()
        self.data_manager.update_product_column(
            self.session_manager.current_results_folder,
            self.data_manager.current_column,
            current_engraving
        )
        
        if self.data_manager.move_to_previous_product():
            prev_engraving = self.data_manager.get_current_engraving()
            self.engraving_number_input.setText(prev_engraving)
            self.statusBar().showMessage(f"제품 컬럼 {self.data_manager.current_column + 1}")
        else:
            QMessageBox.warning(self, "경고", "첫 번째 제품입니다.")
    
    def highlight_current_column(self):
        """현재 컬럼 하이라이트 - GUI 호출용"""
        self.data_manager.highlight_current_column()
    
    def run_ai_analysis(self):
        """AI 분석 실행 - GUI 호출용"""
        if not self.image_manager.is_complete():
            QMessageBox.warning(self, "경고", "9장 이미지가 필요합니다.")
            return
        
        # 테이블 확인 및 생성
        table_path = os.path.join(self.session_manager.current_results_folder, "vertical_object_distances.csv")
        if not os.path.exists(table_path):
            reply = QMessageBox.question(self, "테이블 생성", "분석 테이블을 생성하시겠습니까?")
            if reply == QMessageBox.Yes:
                self.create_table()
            else:
                return
        
        self.session_manager.update_from_ui(self)
        
        results = self.ai_manager.run_ai_analysis(
            self.session_manager.current_capture_folder,
            self.session_manager.current_results_folder
        )
        if results is not None:
            self.data_manager.add_analysis_results(self.session_manager.current_results_folder, results)
            df = pd.read_csv(table_path, encoding='utf-8-sig')
            
            print(f"현재 컬럼: {self.data_manager.current_column}")  # 추가
            print(f"CSV 내용:\n{df.head()}")  # 추가
            
            self.data_manager.display_data_in_gui_table(df, self.session_manager.settings)    
            self.display_test_results()
            self.ai_manager.update_preview_images(self.session_manager.current_capture_folder)
        
    def display_test_results(self):
        """테스트 결과 표시 - GUI 호출용"""
        self.ai_manager.display_test_results(self.session_manager.current_results_folder)
    
    def export_images(self):
        """이미지 내보내기 - GUI 호출용"""
        self.ai_manager.export_images(self.session_manager.current_results_folder)
    
    def generate_excel_report(self):
        """엑셀 보고서 생성 - GUI 호출용"""
        try:
            from excel_report_generator import ExcelReportGenerator
            generator = ExcelReportGenerator(parent=self)
            generator.generate_excel_report()
        except Exception as e:
            QMessageBox.critical(self, "오류", f"보고서 생성 실패: {str(e)}")
    
    def toggle_maximize(self):
        """창 최대화 토글 - GUI 호출용"""
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()
    
    def show_table_context_menu(self, position):
        """테이블 우클릭 메뉴 - GUI 호출용"""
        menu = QMenu(self)
        
        copy_action = menu.addAction("복사")
        copy_action.triggered.connect(self.copy_table_selection)
        
        save_action = menu.addAction("CSV 저장")
        save_action.triggered.connect(self.save_table_as_csv)
        
        menu.exec_(self.distance_table.mapToGlobal(position))
    
    def copy_table_selection(self):
        """테이블 선택 영역 복사"""
        selected = self.distance_table.selectedItems()
        if selected:
            text = "\t".join([item.text() for item in selected])
            QApplication.clipboard().setText(text)
            self.statusBar().showMessage("복사 완료", 2000)
    
    def save_table_as_csv(self):
        """테이블을 CSV로 저장"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "CSV 저장", "", "CSV Files (*.csv)"
        )
        if file_path:
            table = self.distance_table
            with open(file_path, 'w', encoding='utf-8') as f:
                for row in range(table.rowCount()):
                    row_data = []
                    for col in range(table.columnCount()):
                        item = table.item(row, col)
                        row_data.append(item.text() if item else "")
                    f.write(",".join(row_data) + "\n")
            
            self.statusBar().showMessage(f"저장 완료: {file_path}")
    
    def paintEvent(self, event):
        """화면 그리기 - 캡처 영역 표시"""
        painter = QPainter(self)
        
        # 배경
        painter.setCompositionMode(QPainter.CompositionMode_Source)
        painter.setBrush(QColor(240, 240, 240, 255))
        painter.setPen(Qt.NoPen)
        painter.drawRect(self.rect())
        
        # 캡처 영역 투명화
        painter.setCompositionMode(QPainter.CompositionMode_Clear)
        capture_rect = QRect(config.capture_x_coord, config.capture_y_coord,
                           config.capture_width, config.capture_height)
        painter.drawRect(capture_rect)
        
        # 캡처 영역 테두리
        painter.setCompositionMode(QPainter.CompositionMode_Source)
        painter.setPen(QColor(255, 0, 0, 200))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(capture_rect)

def main():
    """메인 함수"""
    QCoreApplication.setAttribute(Qt.AA_UseDesktopOpenGL)
    
    app = QApplication(sys.argv)
    window = SimpleMainProgram()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()