from PySide6.QtWidgets import QMainWindow, QMessageBox
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
import os
import re
from user_manager import UserManager

class RegisterWindow(QMainWindow):
    def __init__(self, login_window):
        super().__init__()
        self.login_window = login_window
        self.user_manager = UserManager()
        self.setup_ui()
        self.setup_icon()
        self.setup_connections()
        
    def setup_ui(self):
        """加载UI文件"""
        loader = QUiLoader()
        ui_file = os.path.join(os.path.dirname(__file__), "ui", "register_window.ui")
        self.ui = loader.load(ui_file)
        self.setCentralWidget(self.ui)
        
        # 设置窗口居中
        self.setGeometry(
            self.screen().geometry().center().x() - self.width() // 2,
            self.screen().geometry().center().y() - self.height() // 2,
            self.width(),
            self.height()
        )
        
        # 设置窗口属性
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint)
        self.setFixedSize(800, 600)
        
        # 设置状态栏信息
        self.ui.statusbar.showMessage("欢迎注册MYYOLO SYSTEM")
        
        # 设置窗口标题
        self.setWindowTitle("YOLO SYSTEM - 注册")
        
    def setup_icon(self):
        # 设置窗口图标
        icon_path = os.path.join(os.path.dirname(__file__), "yolo.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
            self.ui.setWindowIcon(QIcon(icon_path))
        
    def setup_connections(self):
        """连接信号和槽"""
        self.ui.registerButton.clicked.connect(self.register)
        self.ui.backButton.clicked.connect(self.back_to_login)
        
    def register(self):
        """处理注册"""
        username = self.ui.usernameEdit.text()
        password = self.ui.passwordEdit.text()
        confirm_password = self.ui.confirmPasswordEdit.text()
        email = self.ui.emailEdit.text()
        
        if not username or not password or not confirm_password or not email:
            QMessageBox.warning(self, "警告", "请填写所有字段")
            return
            
        if password != confirm_password:
            QMessageBox.warning(self, "错误", "两次输入的密码不一致")
            return
            
        try:
            self.user_manager.register_user(username, password, email)
            QMessageBox.information(self, "成功", "注册成功！")
            self.back_to_login()
        except ValueError as e:
            QMessageBox.warning(self, "错误", str(e))
        
    def back_to_login(self):
        """返回登录窗口"""
        self.login_window.show()
        self.close()
        
    def showEvent(self, event):
        """窗口显示事件"""
        super().showEvent(event)
        # 清空输入框
        self.ui.usernameEdit.clear()
        self.ui.passwordEdit.clear()
        self.ui.confirmPasswordEdit.clear()
        self.ui.emailEdit.clear()
        
    def closeEvent(self, event):
        # 关闭注册窗口时显示登录窗口
        self.login_window.show()
        event.accept() 