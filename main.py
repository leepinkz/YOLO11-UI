import sys
from PySide6.QtWidgets import QApplication
from login import LoginWindow
from main_window import MainWindow

class Application:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.login_window = None
        self.main_window = None
        
    def start(self):
        # 显示登录窗口
        self.show_login()
        return self.app.exec()
        
    def show_login(self):
        self.login_window = LoginWindow()
        # 连接登录成功信号
        self.login_window.login_success.connect(self.on_login_success)
        self.login_window.show()
        
    def on_login_success(self, username):
        # 创建并显示主窗口
        self.main_window = MainWindow()
        self.main_window.show()
        
if __name__ == "__main__":
    app = Application()
    sys.exit(app.start()) 