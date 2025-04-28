from PySide6.QtWidgets import QMainWindow, QMessageBox
from PySide6.QtCore import Signal, Qt
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QIcon
import os
from user_manager import UserManager

class LoginWindow(QMainWindow):
    # 定义登录成功信号
    login_success = Signal(str)  # 传递用户名
    
    def __init__(self):
        super().__init__()
        self.user_manager = UserManager()
        self.setup_ui()
        self.setup_icon()
        self.setup_connections()
        
    def setup_ui(self):
        """加载UI文件"""
        loader = QUiLoader()
        ui_file = os.path.join(os.path.dirname(__file__), "ui", "login_window.ui")
        self.ui = loader.load(ui_file)
        self.setCentralWidget(self.ui)
        
        # 设置窗口属性
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint)
        self.setFixedSize(800, 600)
        
        # 设置状态栏信息
        self.ui.statusbar.showMessage("欢迎使用MYYOLO SYSTEM")
        
        # 设置窗口居中
        self.setGeometry(
            self.screen().geometry().center().x() - self.width() // 2,
            self.screen().geometry().center().y() - self.height() // 2,
            self.width(),
            self.height()
        )
        
        # 设置窗口标题
        self.setWindowTitle("YOLO SYSTEM - 登录")
        
    def setup_icon(self):
        # 设置窗口图标
        icon_path = os.path.join(os.path.dirname(__file__), "yolo.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
            self.ui.setWindowIcon(QIcon(icon_path))
        
    def setup_connections(self):
        """连接信号和槽"""
        self.ui.loginButton.clicked.connect(self.login)
        self.ui.registerButton.clicked.connect(self.show_register)
        
    def login(self):
        username = self.ui.usernameEdit.text()
        password = self.ui.passwordEdit.text()
        
        if not username or not password:
            QMessageBox.warning(self, "警告", "请输入用户名和密码")
            return
            
        if self.user_manager.verify_user(username, password):
            # 发送登录成功信号
            self.login_success.emit(username)
            self.close()
        else:
            QMessageBox.warning(self, "错误", "用户名或密码错误")
            
    def show_register(self):
        from register import RegisterWindow
        self.register_window = RegisterWindow(self)
        self.register_window.show()
        self.hide()
        
    def showEvent(self, event):
        """窗口显示事件"""
        super().showEvent(event)
        # 清空输入框
        self.ui.usernameEdit.clear()
        self.ui.passwordEdit.clear()
        
    def closeEvent(self, event):
        # 直接关闭窗口
        event.accept() 