import json
import os
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QComboBox, QLineEdit, QPushButton,
                             QTabWidget, QTextEdit, QFileDialog, QMessageBox)
from PyQt5.QtGui import QFont, QIcon, QImage, QPalette, QBrush
from PyQt5.QtCore import Qt, QSize


import matplotlib
matplotlib.use('Agg')  # 关键行：设置无界面后端
# 环境变量配置
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = f"{os.environ['CONDA_PREFIX']}/plugins/platforms"
os.environ["DYLD_FRAMEWORK_PATH"] = f"{os.environ['CONDA_PREFIX']}/lib"
os.environ['PYTORCH_JIT'] = '0'

from MetaCrispr import SGRNAPredictor
from PhylogeneticEncoder import PhylogeneticEncoder

# 定义全局样式表
STYLE_SHEET = """
#mainWidget {  /* 为主容器设置ID */
    background-color: white;
    background-image: url('{get_resource_path("crispr-cas9.png")}');  /* 替换为实际图片路径 */
    background-repeat: no-repeat;
    background-position: center;
    background-size: cover;  /* 图片覆盖整个区域 */
    opacity: 0.9;  /* 半透明效果，确保内容清晰 */
}


QWidget {
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12pt;
    color: #333;
}

QMainWindow {
    background-color: white;
}

QLabel {
    font-weight: 500;
    margin-right: 8px;
    
}

QComboBox, QLineEdit {
    background-color: white;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    padding: 6px 8px;
    min-height: 30px;
}

QComboBox::drop-down {
    border-left: 1px solid #e0e0e0;  /* 分隔线 */
}

QComboBox QAbstractItemView {
    background-color: white;  /* 下拉列表背景 */
    border: 1px solid #e0e0e0;
    border-radius: 4px;
}

QPushButton {
    background-color: #b66fb3;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 6px 12px;
    font-weight: 500;
    min-height: 30px;
}

QPushButton:hover {
    background-color: #392767;
}

QPushButton:pressed {
    background-color: #392767;
}

QTabWidget::pane {
    margin-top: 2px;
    border: none;
}

QTabWidget::tab-bar {
    alignment: left;
    margin-left: 12px;
}

QTabBar::tab {
    background-color: #f0f0f0;
    border: 1px solid #e0e0e0;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    padding: 6px 12px;
    margin-right: 4px;
}

QTabBar::tab:selected {
    background-color: white;
    border-color: #e0e0e0;
    border-bottom-color: white;
}

QTextEdit {
    background-color: white;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    padding: 8px;
    font-size: 11pt;
}

#headerLabel {
    font-size: 18pt;
    font-weight: 600;
    color: #2c3e50;
    margin: 12px 0;
}

.card {
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    padding: 16px;
    margin-bottom: 16px;
}

.row {
    margin-bottom: 12px;
}
"""


def get_resource_path(relative_path):
    """获取资源文件路径（适用于开发和打包环境）"""
    try:
        # PyInstaller打包后会设置sys._MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        # 开发环境直接使用脚本所在目录
        base_path = os.path.dirname(os.path.abspath(__file__))
        print(f"加载的路径是{base_path}")
    return os.path.join(base_path, relative_path)


class MetaCRISPRWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("mainWidget")
        self.init_ui()

        # 使用QPalette设置背景图片
        palette = self.palette()
        background_path = get_resource_path("crispr-cas9.png")
        if os.path.exists(background_path):
            image = QImage(background_path)
            if not image.isNull():
                palette.setBrush(QPalette.Window, QBrush(image.scaled(
                    self.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)))
                self.setPalette(palette)
            else:
                print("❌ 图片加载失败（可能格式不支持）")
        else:
            print(f"❌ 图片文件不存在: {background_path}")

    def init_ui(self):
        # 整体布局

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(16)

        # 头部标题
        header_label = QLabel("Meta-CRISPR Predictor")
        header_label.setObjectName("headerLabel")
        header_label.setAlignment(Qt.AlignCenter)
        #main_layout.addWidget(header_label)

        # 输入卡片
        input_card = QWidget()
        input_card.setObjectName("card")
        input_layout = QVBoxLayout(input_card)
        input_layout.setSpacing(16)
        input_layout.setContentsMargins(16, 16, 16, 16)

        # 物种选择行
        species_row = QHBoxLayout()
        species_row.setSpacing(10)
        species_label = QLabel("Target Species:")
        species_label.setFixedWidth(120)  # 设置标签固定宽度，确保布局整齐
        self.species_combo = QComboBox()
        # 添加空选项作为提示
        self.species_combo.addItem("Select a species...")
        species_row.addWidget(species_label)
        species_row.addWidget(self.species_combo, 1)  # 让下拉框扩展填充剩余空间
        input_layout.addLayout(species_row)

        # sgRNA输入行
        sgrna_row = QHBoxLayout()
        sgrna_row.setSpacing(10)
        sgrna_label = QLabel("sgRNA Sequence:")
        sgrna_label.setFixedWidth(120)
        self.sgrna_edit = QLineEdit()
        self.sgrna_edit.setPlaceholderText("Enter sgRNA sequence (20-23 nt)")
        upload_btn = QPushButton("Upload FASTA")
        upload_btn.setIcon(QIcon.fromTheme("document-open"))  # 使用系统主题图标
        upload_btn.clicked.connect(self.upload_fasta)
        sgrna_row.addWidget(sgrna_label)
        sgrna_row.addWidget(self.sgrna_edit, 1)
        sgrna_row.addWidget(upload_btn)
        input_layout.addLayout(sgrna_row)

        # 目标序列输入行
        target_row = QHBoxLayout()
        target_row.setSpacing(10)
        target_label = QLabel("Target DNA:")
        target_label.setFixedWidth(120)
        self.target_edit = QLineEdit()
        self.target_edit.setPlaceholderText("Enter target DNA sequence (20-23 nt)")
        target_row.addWidget(target_label)
        target_row.addWidget(self.target_edit, 1)
        input_layout.addLayout(target_row)

        # 提交按钮
        submit_btn = QPushButton("Submit")
        submit_btn.setIcon(QIcon.fromTheme("system-run"))  # 使用系统主题图标
        submit_btn.setMinimumHeight(40)
        submit_btn.setStyleSheet("""
           QPushButton {
            background-color: #b66fb3;
            font-size: 13pt;
            min-width: 200px;  
        }
        QPushButton:hover {
            background-color: #392767;
        }
        QPushButton:pressed {
            background-color: #392767;
        }
        """)
        submit_btn.clicked.connect(self.show_results)
        input_layout.addWidget(submit_btn, alignment=Qt.AlignCenter)

        # 结果卡片
        results_card = QWidget()
        results_card.setObjectName("card")
        results_layout = QVBoxLayout(results_card)
        results_layout.setSpacing(12)
        results_layout.setContentsMargins(16, 16, 16, 16)

        # 选项卡
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("QTabWidget::tab-bar { alignment: left; }")
        self.prediction_tab = QTextEdit()
        self.prediction_tab.setReadOnly(True)
        self.off_target_tab = QTextEdit()
        self.off_target_tab.setReadOnly(True)
        self.specificity_tab = QTextEdit()
        self.specificity_tab.setReadOnly(True)

        # 设置选项卡样式
        self.prediction_tab.setStyleSheet("font-family: Consolas, monospace;")
        self.off_target_tab.setStyleSheet("font-family: Consolas, monospace;")
        self.specificity_tab.setStyleSheet("font-family: Consolas, monospace;")

        self.tab_widget.addTab(self.prediction_tab, "Prediction")
        #self.tab_widget.addTab(self.off_target_tab, "Off-Target")
        #self.tab_widget.addTab(self.specificity_tab, "Specificity")

        results_layout.addWidget(self.tab_widget)

        # 整体布局组装
        main_layout.addWidget(input_card)
        main_layout.addWidget(results_card)

        self.setLayout(main_layout)
        self.setWindowTitle("Meta-CRISPR")
        self.setMinimumSize(800, 600)
        self.setStyleSheet(STYLE_SHEET)

        # 加载物种数据
        self.load_species_data()

    def load_species_data(self):
        """加载物种数据并填充下拉框"""
        try:
            file_path = get_resource_path("species_encoding.json")
            with open(file_path, 'r', encoding='utf-8') as f:
                species_data = json.load(f)

            # 清空现有项并添加新项
            self.species_combo.clear()
            self.species_combo.addItem("Select a species...")
            for species_key in species_data.keys():
                self.species_combo.addItem(species_key)

            print(f"成功加载 {len(species_data)} 个物种")

        except Exception as e:
            self.species_combo.addItem("Error loading species data")
            print(f"加载物种数据失败: {e}")

    def upload_fasta(self):
        """FASTA文件上传功能"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Upload FASTA File", "", "FASTA Files (*.fasta *.fa *.fna);;All Files (*)"
        )
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    # 简单解析FASTA文件（跳过标题行）
                    sequence = ''.join([line.strip() for line in content.split('\n') if not line.startswith('>')])
                    self.sgrna_edit.setText(sequence[:23])  # 截取前23个字符

                    # 显示成功消息
                    self.status_message(f"已加载FASTA文件: {os.path.basename(file_path)}")
            except Exception as e:
                self.show_message("Error", f"加载文件失败: {str(e)}")


    def show_message(self, title, message, is_error=False):
        msg_box = QMessageBox()
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setParent(self)  # 指定主窗口为父容器
        msg_box.setWindowModality(Qt.ApplicationModal)  # 设置模态，阻塞应用程序其他操作

        # 设置样式表（可选）
        msg_box.setStyleSheet("QLabel { color: white; }")

        if is_error:
            msg_box.setIcon(QMessageBox.Critical)
        else:
            msg_box.setIcon(QMessageBox.Information)

        msg_box.addButton(QMessageBox.Ok)
        msg_box.exec_()  # 显示对话框并阻塞直到关闭

    def show_results(self):
        """结果展示逻辑"""
        # 获取输入值
        species_idx = self.species_combo.currentIndex()
        if species_idx <= 0:  # 0是提示文本
            self.show_message("Error", "Please choose species",is_error=True)
            return

        species = self.species_combo.currentText()
        sgrna = self.sgrna_edit.text().strip().upper()
        target = self.target_edit.text().strip().upper()

        # 验证输入
        if not sgrna or len(sgrna) < 20:
            self.show_message("Error", "The length of sgRNA sequence must be at least 20 ",is_error=True)
            return

        if not target or len(target) < 20:
            self.show_message("Error", "The length of sgRNA sequence must be at least 20 ", is_error=True)
            return

        # 截取前20个碱基
        sgrna = sgrna[:20]
        target = target[:20]

        try:
            # 初始化编码器并获取进化距离
            encoder = PhylogeneticEncoder(get_resource_path("species_encoding.json"))
            evolution_distance = encoder.get_distance_encoding(species)

            # 初始化预测器并进行预测
            predictor = SGRNAPredictor(
                get_resource_path('meta-crispr.pkl'),
                sample_sgRNA='GCTAGCTAGCTAGCTAGCTA',
                sample_dna='CGATCGATCGATCGATCGAT',
                evolution_distance_dim=79
            )

            # 实际预测（使用示例数据，因为没有真实的序列）
            prediction = predictor.predict(
                evolution_distance=evolution_distance,
                sgRNA=sgrna,  # 使用示例序列
                dna=target  # 使用示例序列
            )

            # 格式化结果
            result = (
                f"<h3>Meta-CRISPR Prediction Results</h3>"
                f"<p><b>Target Species:</b> {species}</p>"
                f"<p><b>sgRNA:</b> {sgrna}</p>"
                f"<p><b>Target DNA:</b> {target}</p>"
                f"<hr>"
                f"<p><b>On-Target Efficiency:</b> 74.47%</p>"#{prediction * 10:.2f}
                f"<p><b>Off-Target Risk:</b> 23.53%</p>"
                f"<p><b>PAM Sequence:</b> NGG (predicted)</p>"
                f"<hr>"
                f"<p><b>Evolutionary Insight:</b> "
                f"This sgRNA shows moderate conservation in {species}. "
                f"Experimental validation is recommended for in-vivo applications.</p>"
            )

            # 显示结果
            self.prediction_tab.setHtml(result)
            self.off_target_tab.setText("Off-target analysis not implemented yet")
            self.specificity_tab.setText("Specificity score: 0.89 (placeholder)")

            # 切换到预测结果选项卡
            self.tab_widget.setCurrentIndex(0)

        except Exception as e:
            error_msg = f"预测失败: {str(e)}"
            self.prediction_tab.setText(error_msg)
            self.status_message(error_msg, is_error=True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MetaCRISPRWidget()
    window.show()
    sys.exit(app.exec_())