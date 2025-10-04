import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, QCheckBox, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PIL import Image
import torch
import numpy as np


class YOLOApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # 初始化变量
        self.image_path = None
        self.result_image = None

        # 加载模型
        try:
            self.model = torch.hub.load("ultralytics/yolov5", "custom",
                                        path="runs/train/exp5/weights/best.pt")
            print("模型加载成功")
        except Exception as e:
            print(f"模型加载失败: {e}")
            self.model = None

        # 初始化UI
        self.initUI()

    def initUI(self):
        # 主窗口设置
        self.setWindowTitle('YOLOv5 抬头检测')
        self.setGeometry(100, 100, 1200, 600)

        # 主窗口部件
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # 图片显示区域
        image_layout = QHBoxLayout()

        # 左侧上传区域
        left_widget = QWidget()
        left_layout = QVBoxLayout()

        self.left_frame = QLabel("点击上传图片")
        self.left_frame.setAlignment(Qt.AlignCenter)
        self.left_frame.setMinimumSize(500, 400)
        self.left_frame.setStyleSheet("border: 2px dashed gray; background-color: #f0f0f0;")
        self.left_frame.mousePressEvent = self.upload_image

        left_layout.addWidget(self.left_frame)
        left_widget.setLayout(left_layout)

        # 右侧结果区域
        right_widget = QWidget()
        right_layout = QVBoxLayout()

        self.right_frame = QLabel("检测结果将显示在这里")
        self.right_frame.setAlignment(Qt.AlignCenter)
        self.right_frame.setMinimumSize(500, 400)
        self.right_frame.setStyleSheet("border: 2px dashed gray; background-color: #f0f0f0;")

        right_layout.addWidget(self.right_frame)
        right_widget.setLayout(right_layout)

        image_layout.addWidget(left_widget)
        image_layout.addWidget(right_widget)

        # 按钮区域
        button_layout = QHBoxLayout()

        self.generate_btn = QPushButton("生成检测结果")
        self.generate_btn.clicked.connect(self.generate_result)
        self.generate_btn.setEnabled(False)  # 初始不可用

        self.save_checkbox = QCheckBox("保存检测结果")

        button_layout.addWidget(self.generate_btn)
        button_layout.addWidget(self.save_checkbox)
        button_layout.addStretch()

        # 添加到主布局
        main_layout.addLayout(image_layout)
        main_layout.addLayout(button_layout)
        main_widget.setLayout(main_layout)

        self.setCentralWidget(main_widget)

    def upload_image(self, event):
        # 打开文件对话框选择图片
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp)"
        )

        if file_path:
            self.image_path = file_path
            # 显示原始图片
            pixmap = QPixmap(file_path)
            scaled_pixmap = pixmap.scaled(
                self.left_frame.width(),
                self.left_frame.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.left_frame.setPixmap(scaled_pixmap)
            self.generate_btn.setEnabled(True)  # 允许生成结果

    def generate_result(self):
        if not self.image_path or not self.model:
            return

        try:
            # 使用模型检测
            img = Image.open(self.image_path)
            results = self.model(img)

            # 渲染检测结果
            results.render()  # 渲染检测结果

            # 获取渲染后的图像（第一个图像）
            im_array = results.ims[0] if results.ims and len(results.ims) > 0 else None

            if im_array is not None:
                # 转换为QImage
                height, width, channel = im_array.shape
                bytes_per_line = 3 * width
                q_img = QImage(im_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)

                # 显示结果
                scaled_pixmap = pixmap.scaled(
                    self.right_frame.width(),
                    self.right_frame.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.right_frame.setPixmap(scaled_pixmap)

                # 保存渲染后的图像到临时变量，以便保存功能使用
                self.result_image = im_array

                # 如果勾选了保存选项，则保存图片
                if self.save_checkbox.isChecked():
                    self.save_image()
            else:
                QMessageBox.warning(self, "错误", "未能生成检测结果图像")

        except Exception as e:
            print(f"生成检测结果时出错: {e}")
            QMessageBox.critical(self, "错误", f"生成检测结果时出错: {e}")

    def save_image(self):
        if self.result_image is None:
            QMessageBox.warning(self, "警告", "没有可保存的图像，请先生成检测结果")
            return

        # 打开文件对话框选择保存位置
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存检测结果",
            os.path.join(os.path.expanduser("~"), "Desktop", "检测结果.jpg"),
            "图片文件 (*.jpg *.png)"
        )

        if file_path:
            try:
                # 使用PIL保存图像
                img = Image.fromarray(self.result_image)
                img.save(file_path)
                QMessageBox.information(self, "成功", f"图片已保存到: {file_path}")
            except Exception as e:
                print(f"保存图片时出错: {e}")
                QMessageBox.critical(self, "错误", f"保存图片时出错: {e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = YOLOApp()
    ex.show()
    sys.exit(app.exec_())