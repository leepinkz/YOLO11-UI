import json
import os
from typing import Dict, Optional

class UserManager:
    def __init__(self, data_file: str = "users.json"):
        self.data_file = data_file
        self.users: Dict[str, dict] = {}
        self.load_users()

    def load_users(self) -> None:
        """从JSON文件加载用户数据"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    self.users = json.load(f)
            except json.JSONDecodeError:
                self.users = {}
        else:
            self.users = {}

    def save_users(self) -> None:
        """保存用户数据到JSON文件"""
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(self.users, f, ensure_ascii=False, indent=4)

    def register_user(self, username: str, password: str, email: str) -> bool:
        """注册新用户"""
        if username in self.users:
            return False  # 用户名已存在
        
        self.users[username] = {
            'password': password,
            'email': email
        }
        self.save_users()
        return True

    def verify_user(self, username: str, password: str) -> bool:
        """验证用户登录"""
        if username not in self.users:
            return False
        
        return self.users[username]['password'] == password

    def get_user_info(self, username: str) -> Optional[dict]:
        """获取用户信息"""
        return self.users.get(username)

    def update_user_info(self, username: str, **kwargs) -> bool:
        """更新用户信息"""
        if username not in self.users:
            return False
        
        for key, value in kwargs.items():
            if key in self.users[username]:
                self.users[username][key] = value
        
        self.save_users()
        return True

    def delete_user(self, username: str) -> bool:
        """删除用户"""
        if username not in self.users:
            return False
        
        del self.users[username]
        self.save_users()
        return True 