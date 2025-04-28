from PySide6.QtWidgets import QMainWindow, QFileDialog, QMessageBox, QMenu, QInputDialog
from PySide6.QtCore import Qt, QTimer
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import QStyle
import cv2
import numpy as np
from yolo_detector import YOLODetector
import time
import os
import glob

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 设置窗口标题
        self.setWindowTitle("MYYOLO SYSTEM")
        
        # 设置窗口图标
        icon_path = os.path.join(os.path.dirname(__file__), "yolo.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        # 设置状态栏信息
        self.statusBar().showMessage("Author: leepinkz | Email: zz17373248564@163.com | Version: 1.0")
        
        # 加载UI文件
        loader = QUiLoader()
        self.ui = loader.load("ui/main_window.ui")
        self.setCentralWidget(self.ui)
        
        # 初始化YOLO检测器
        self.detector = None
        self.current_image = None
        self.detection_timer = QTimer()
        self.detection_timer.timeout.connect(self.process_detection)
        
        # 初始化进度条动画
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_value = 0
        self.progress_direction = 1
        
        # 设置菜单图标
        self.setup_menu_icons()
        
        # 设置消息框右键菜单
        self.setup_message_context_menu()
        
        # 连接信号和槽
        self.connect_signals()
        
        # 初始化UI状态
        self.init_ui_state()
        
        # 初始化变量
        self.camera_timer = None
        self.camera_cap = None
        self.video_path = None
        self.video_writer = None
        
    def setup_menu_icons(self):
        """设置菜单图标"""
        # 创建图标字典，使用自定义图标
        icon_dict = {
            "图片检测": "icons/image.png",  # 图片检测图标
            "视频检测": "icons/video.png",  # 视频检测图标
            "摄像头检测": "icons/camera.png",  # 摄像头检测图标
            "批量处理": "icons/batch.png",  # 批量处理图标
            "数据库": "icons/database.png",  # 数据库图标
        }
        
        # 为每个菜单项设置图标
        for i in range(self.ui.menuList.count()):
            item = self.ui.menuList.item(i)
            text = item.text()
            if text in icon_dict:
                # 获取图标路径
                icon_path = icon_dict[text]
                if os.path.exists(icon_path):
                    # 如果图标文件存在，使用自定义图标
                    icon = QIcon(icon_path)
                else:
                    # 如果图标文件不存在，使用Qt标准图标作为备用
                    standard_icons = {
                        "图片检测": QStyle.StandardPixmap.SP_FileIcon,
                        "视频检测": QStyle.StandardPixmap.SP_MediaPlay,
                        "摄像头检测": QStyle.StandardPixmap.SP_ComputerIcon,
                        "批量处理": QStyle.StandardPixmap.SP_FileDialogDetailedView,
                        "数据库": QStyle.StandardPixmap.SP_DirIcon,
                    }
                    icon = self.style().standardIcon(standard_icons[text])
                
                # 创建更大的图标
                pixmap = icon.pixmap(32, 32)  # 设置图标大小为32x32
                larger_icon = QIcon(pixmap)
                item.setIcon(larger_icon)
                
    def setup_message_context_menu(self):
        """设置消息框右键菜单"""
        # 创建右键菜单
        self.message_menu = QMenu(self.ui.messageOutput)
        
        # 添加复制选项
        copy_action = self.message_menu.addAction("复制")
        copy_action.triggered.connect(self.copy_message)
        
        # 添加分隔线
        self.message_menu.addSeparator()
        
        # 添加清除所有选项
        clear_action = self.message_menu.addAction("清除所有")
        clear_action.triggered.connect(self.clear_messages)
        
        # 设置消息框的右键菜单策略
        self.ui.messageOutput.setContextMenuPolicy(Qt.CustomContextMenu)
        self.ui.messageOutput.customContextMenuRequested.connect(self.show_message_context_menu)
        
    def show_message_context_menu(self, pos):
        """显示消息框右键菜单"""
        self.message_menu.exec(self.ui.messageOutput.mapToGlobal(pos))
        
    def clear_messages(self):
        """清除所有消息"""
        self.ui.messageOutput.clear()
        self.append_message("消息已清除")
        
    def copy_message(self):
        """复制选中的消息"""
        self.ui.messageOutput.copy()
        
    def connect_signals(self):
        """连接UI组件的信号和槽"""
        # 菜单项点击
        self.ui.menuList.itemClicked.connect(self.menu_item_clicked)
        
        # 控制按钮
        self.ui.startButton.clicked.connect(self.start_detection)
        self.ui.stopButton.clicked.connect(self.stop_detection)
        
        # 参数设置
        self.ui.confidenceSlider.valueChanged.connect(self.update_confidence)
        self.ui.iouSlider.valueChanged.connect(self.update_iou)
        self.ui.lineWidthSlider.valueChanged.connect(self.update_line_width)
        self.ui.browseModelButton.clicked.connect(self.browse_model_path)
        self.ui.saveCheckBox.stateChanged.connect(self.save_state_changed)
        self.ui.browseButton.clicked.connect(self.browse_save_path)
        
        # 数值输入框
        self.ui.confidenceSpinBox.valueChanged.connect(self.update_confidence_from_spinbox)
        self.ui.iouSpinBox.valueChanged.connect(self.update_iou_from_spinbox)
        self.ui.lineWidthSpinBox.valueChanged.connect(self.update_line_width_from_spinbox)
        
    def init_ui_state(self):
        """初始化UI状态"""
        # 设置初始值
        self.ui.confidenceSpinBox.setValue(0.5)
        self.ui.iouSpinBox.setValue(0.5)
        self.ui.lineWidthSpinBox.setValue(3)
        
        # 更新标签显示
        self.update_confidence(50)  # 滑动条仍然使用0-100的值
        self.update_iou(50)        # 滑动条仍然使用0-100的值
        self.update_line_width(3)
        
    def browse_model_path(self):
        """选择模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择模型文件",
            "",
            "模型文件 (*.pt)"
        )
        
        if file_path:
            self.ui.modelPathEdit.setText(file_path)
            self.append_message(f"已选择模型: {file_path}")
            
    def menu_item_clicked(self, item):
        """处理菜单项点击"""
        text = item.text()
        
        # 获取模型路径
        model_path = self.ui.modelPathEdit.text()
        if not model_path:
            self.append_message("请先选择模型文件")
            return
            
        # 加载模型
        try:
            self.detector = YOLODetector(model_path)
            self.append_message(f"已加载模型: {model_path}")
            
            # 设置检测参数
            confidence = self.ui.confidenceSpinBox.value()
            iou = self.ui.iouSpinBox.value()
            line_width = self.ui.lineWidthSpinBox.value()
            
            self.detector.confidence = confidence
            self.detector.iou = iou
            self.detector.line_width = line_width
            
        except Exception as e:
            self.append_message(f"加载模型失败: {str(e)}")
            return
            
        if text == "图片检测":
            self.select_image()
        elif text == "视频检测":
            self.select_video()
        elif text == "摄像头检测":
            # 直接启动摄像头检测
            self.start_camera()
        elif text == "批量处理":
            self.batch_process()
        elif text == "数据库":
            self.show_database()
        elif text == "姿态检测":
            # 启动摄像头检测
            self.start_camera()
            
    def select_image(self):
        """选择图片文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp)"
        )
        
        if file_path:
            self.current_image = cv2.imread(file_path)
            if self.current_image is not None:
                self.display_image(self.current_image, self.ui.inputLabel)
                self.append_message(f"已加载图片: {file_path}")
                self.start_detection()
            else:
                self.append_message("无法加载图片")
                
    def display_image(self, image, label):
        """显示图片到指定标签"""
        if image is None:
            return
            
        # 调整图片大小以适应标签
        height, width = image.shape[:2]
        label_size = label.size()
        scale = min(label_size.width() / width, label_size.height() / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # 调整图片大小
        resized = cv2.resize(image, (new_width, new_height))
        
        # 转换颜色空间
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 转换为QImage
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # 显示图片
        label.setPixmap(QPixmap.fromImage(qt_image))
        
    def update_progress(self):
        """更新进度条动画"""
        if self.progress_direction == 1:
            self.progress_value += 5
            if self.progress_value >= 100:
                self.progress_value = 100
                self.progress_direction = -1
        else:
            self.progress_value -= 5
            if self.progress_value <= 0:
                self.progress_value = 0
                self.progress_direction = 1
                
        self.ui.progressBar.setValue(self.progress_value)
        
    def start_progress_animation(self):
        """开始进度条动画"""
        self.progress_value = 0
        self.progress_direction = 1
        self.ui.progressBar.setValue(0)
        self.progress_timer.start(100)  # 每100ms更新一次
        
    def stop_progress_animation(self):
        """停止进度条动画"""
        self.progress_timer.stop()
        self.ui.progressBar.setValue(0)
        
    def start_detection(self):
        """开始检测"""
        if self.detector is None:
            self.append_message("请先加载模型")
            return
            
        # 如果正在使用摄像头或视频，直接返回
        if self.camera_timer and self.camera_timer.isActive():
            return
            
        # 如果是图片检测
        if self.current_image is not None:
            try:
                # 执行检测
                result_image, detections = self.detector.detect(self.current_image)
                
                # 显示结果
                self.display_image(result_image, self.ui.outputLabel)
                
                # 显示检测信息
                for det in detections:
                    self.append_message(f"检测到: {det['class']}, 置信度: {det['confidence']:.2f}")
                    
                # 如果启用了保存结果
                if self.ui.saveCheckBox.isChecked():
                    save_path = self.ui.savePathEdit.text()
                    if save_path:
                        # 生成文件名（使用时间戳）
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"detection_{timestamp}.jpg"
                        full_path = os.path.join(save_path, filename)
                        
                        # 保存结果图片
                        cv2.imwrite(full_path, result_image)
                        self.append_message(f"结果已保存到: {full_path}")
                    else:
                        self.append_message("请先选择保存路径")
                        
                self.append_message("检测完成")
                
            except Exception as e:
                self.append_message(f"检测失败: {str(e)}")
                
        # 如果是视频检测
        elif hasattr(self, 'video_path') and self.video_path:
            self.start_video(self.video_path)
            
        # 如果是摄像头检测
        else:
            self.start_camera()
            
    def process_detection(self):
        """处理检测"""
        if self.current_image is None or self.detector is None:
            self.detection_timer.stop()
            return
            
        try:
            # 执行检测
            result_image, detections = self.detector.detect(self.current_image)
            
            # 显示结果
            self.display_image(result_image, self.ui.outputLabel)
            
            # 显示检测信息
            for det in detections:
                self.append_message(f"检测到: {det['class']}, 置信度: {det['confidence']:.2f}")
                
            # 停止检测
            self.detection_timer.stop()
            self.append_message("检测完成")
            
        except Exception as e:
            self.append_message(f"检测失败: {str(e)}")
            self.detection_timer.stop()
            
    def stop_detection(self):
        """停止检测"""
        # 停止进度条动画
        self.stop_progress_animation()
        
        # 停止定时器
        if self.camera_timer:
            self.camera_timer.stop()
            self.camera_timer = None
            
        # 释放摄像头/视频
        if self.camera_cap:
            self.camera_cap.release()
            self.camera_cap = None
            
        # 关闭视频写入器
        if hasattr(self, 'video_writer') and self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            
        # 清空输入输出图像
        self.ui.inputLabel.clear()
        self.ui.outputLabel.clear()
        
        # 更新状态
        self.append_message("检测已停止")
        self.statusBar().showMessage("就绪")
        
    def update_confidence(self, value):
        """更新置信度阈值"""
        # 将滑动条的0-100值转换为0-1
        confidence = value / 100.0
        self.ui.confidenceSpinBox.setValue(confidence)
        self.ui.confidenceValueLabel.setText(f"{confidence:.2f}")
        
    def update_iou(self, value):
        """更新IOU阈值"""
        # 将滑动条的0-100值转换为0-1
        iou = value / 100.0
        self.ui.iouSpinBox.setValue(iou)
        self.ui.iouValueLabel.setText(f"{iou:.2f}")
        
    def update_line_width(self, value):
        """更新线宽"""
        self.ui.lineWidthSpinBox.setValue(value)
        self.ui.lineWidthValueLabel.setText(str(value))
        
    def update_confidence_from_spinbox(self, value):
        """从SpinBox更新置信度"""
        # 将0-1的值转换为滑动条的0-100
        slider_value = int(value * 100)
        self.ui.confidenceSlider.setValue(slider_value)
        self.ui.confidenceValueLabel.setText(f"{value:.2f}")
        
    def update_iou_from_spinbox(self, value):
        """从SpinBox更新IOU"""
        # 将0-1的值转换为滑动条的0-100
        slider_value = int(value * 100)
        self.ui.iouSlider.setValue(slider_value)
        self.ui.iouValueLabel.setText(f"{value:.2f}")
        
    def update_line_width_from_spinbox(self, value):
        """从SpinBox更新线宽"""
        self.ui.lineWidthSlider.setValue(value)
        self.ui.lineWidthValueLabel.setText(str(value))
        
    def save_state_changed(self, state):
        """保存选项状态改变"""
        if state == 2:  # 2 表示选中状态
            self.append_message("已启用保存结果")
        else:  # 0 表示未选中状态
            self.append_message("已禁用保存结果")
            
    def browse_save_path(self):
        """选择保存路径"""
        # 获取当前路径（如果有）
        current_path = self.ui.savePathEdit.text()
        if not current_path:
            current_path = "."
            
        # 打开文件夹选择对话框
        save_path = QFileDialog.getExistingDirectory(
            self,
            "选择保存位置",
            current_path,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        # 如果用户选择了路径
        if save_path:
            self.ui.savePathEdit.setText(save_path)
            self.append_message(f"保存路径已设置为: {save_path}")
            
    def append_message(self, message):
        """添加消息到消息框"""
        self.ui.messageOutput.append(message)
        
    def start_camera(self):
        """启动摄像头检测"""
        if self.detector is None:
            self.append_message("请先加载模型")
            return
            
        try:
            # 获取可用的摄像头列表
            available_cameras = YOLODetector.get_available_cameras()
            if not available_cameras:
                self.append_message("未找到可用的摄像头")
                return
                
            # 如果有多个摄像头，让用户选择
            if len(available_cameras) > 1:
                camera_id, ok = QInputDialog.getItem(
                    self, "选择摄像头", "请选择要使用的摄像头:",
                    [f"摄像头 {i}" for i in available_cameras], 0, False)
                if not ok:
                    return
                camera_id = available_cameras[int(camera_id.split()[-1])]
            else:
                camera_id = available_cameras[0]
                
            # 打开摄像头
            self.camera_cap = cv2.VideoCapture(camera_id)
            if not self.camera_cap.isOpened():
                self.append_message(f"无法打开摄像头 ID: {camera_id}")
                return
                
            # 获取摄像头信息
            width = int(self.camera_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.camera_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.camera_cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:  # 某些摄像头可能会返回0的fps
                fps = 30
                
            # 如果需要保存结果
            if self.ui.saveCheckBox.isChecked():
                save_path = self.ui.savePathEdit.text()
                if save_path:
                    # 生成输出视频文件名
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    output_filename = f"camera_{timestamp}.mp4"
                    output_path = os.path.join(save_path, output_filename)
                    
                    # 创建视频写入器
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    self.append_message(f"结果将保存到: {output_path}")
                else:
                    self.append_message("请先选择保存路径")
                    self.video_writer = None
            else:
                self.video_writer = None
                
            # 创建定时器
            self.camera_timer = QTimer()
            self.camera_timer.timeout.connect(self.process_camera_frame)
            self.camera_timer.start(30)  # 约30fps
            
            self.append_message(f"已启动摄像头 ID: {camera_id}")
            self.statusBar().showMessage(f"摄像头运行中 (ID: {camera_id})")
            
        except Exception as e:
            self.append_message(f"启动摄像头失败: {str(e)}")
            
    def process_camera_frame(self):
        """处理摄像头帧"""
        if self.camera_cap is None or not self.camera_cap.isOpened():
            return
            
        ret, frame = self.camera_cap.read()
        if not ret:
            return
            
        try:
            # 显示输入图像
            self.display_image(frame, self.ui.inputLabel)
            
            # 执行检测
            result_frame, _ = self.detector.detect(frame)
            
            # 显示输出图像
            self.display_image(result_frame, self.ui.outputLabel)
            
            # 如果需要保存结果
            if self.video_writer is not None:
                self.video_writer.write(result_frame)
            
        except Exception as e:
            self.append_message(f"处理摄像头帧失败: {str(e)}")
            
    def select_video(self):
        """选择视频文件"""
        video_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            "",
            "视频文件 (*.mp4 *.avi *.mov)"
        )
        
        if video_path:
            self.start_video(video_path)
            
    def start_video(self, video_path):
        """启动视频检测"""
        if self.detector is None:
            self.append_message("请先加载模型")
            return
            
        try:
            # 保存视频路径
            self.video_path = video_path
            
            # 打开视频文件
            self.camera_cap = cv2.VideoCapture(video_path)
            if not self.camera_cap.isOpened():
                self.append_message(f"无法打开视频文件: {video_path}")
                return
                
            # 获取视频信息
            width = int(self.camera_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.camera_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.camera_cap.get(cv2.CAP_PROP_FPS)
            
            # 如果需要保存结果
            if self.ui.saveCheckBox.isChecked():
                save_path = self.ui.savePathEdit.text()
                if save_path:
                    # 生成输出视频文件名
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    output_filename = f"detection_{timestamp}.mp4"
                    output_path = os.path.join(save_path, output_filename)
                    
                    # 创建视频写入器
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    self.append_message(f"结果将保存到: {output_path}")
                else:
                    self.append_message("请先选择保存路径")
                    self.video_writer = None
            else:
                self.video_writer = None
                
            # 创建定时器
            self.camera_timer = QTimer()
            self.camera_timer.timeout.connect(self.process_camera_frame)
            self.camera_timer.start(30)  # 约30fps
            
            self.append_message(f"已加载视频: {video_path}")
            self.statusBar().showMessage(f"视频播放中: {os.path.basename(video_path)}")
            
        except Exception as e:
            self.append_message(f"加载视频失败: {str(e)}")
            
    def batch_process(self):
        """批量处理"""
        if self.detector is None:
            self.append_message("请先加载模型")
            return
            
        # 选择输入文件夹
        input_dir = QFileDialog.getExistingDirectory(
            self,
            "选择输入文件夹",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if not input_dir:
            return
            
        # 选择输出文件夹
        output_dir = QFileDialog.getExistingDirectory(
            self,
            "选择输出文件夹",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if not output_dir:
            return
            
        # 获取所有图片文件
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(glob.glob(os.path.join(input_dir, ext)))
            
        if not image_files:
            self.append_message("未找到图片文件")
            return
            
        # 设置进度条
        self.ui.progressBar.setMaximum(len(image_files))
        self.ui.progressBar.setValue(0)
        
        # 开始批量处理
        self.append_message(f"开始批量处理，共 {len(image_files)} 张图片")
        
        for i, image_path in enumerate(image_files):
            try:
                # 读取图片
                image = cv2.imread(image_path)
                if image is None:
                    self.append_message(f"无法读取图片: {image_path}")
                    continue
                    
                # 执行检测
                result_image, detections = self.detector.detect(image)
                
                # 保存结果
                output_path = os.path.join(output_dir, os.path.basename(image_path))
                cv2.imwrite(output_path, result_image)
                
                # 更新进度条
                self.ui.progressBar.setValue(i + 1)
                
                # 显示处理信息
                self.append_message(f"已处理: {os.path.basename(image_path)}")
                for det in detections:
                    self.append_message(f"检测到: {det['class']}, 置信度: {det['confidence']:.2f}")
                    
            except Exception as e:
                self.append_message(f"处理图片 {image_path} 失败: {str(e)}")
                
        self.append_message("批量处理完成")
        self.ui.progressBar.setValue(0)
        
    def closeEvent(self, event):
        """窗口关闭事件"""
        # 停止检测
        self.stop_detection()
        event.accept() 