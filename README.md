# YOLO目标检测系统

这是一个基于YOLO11的图形界面目标检测系统，支持多种检测任务，包括目标检测、实例分割、姿态估计和图像分类。

## 功能特点

- 支持多种YOLO任务：目标检测、实例分割、姿态估计、图像分类
- 用户友好的图形界面
- 支持图片、视频和摄像头实时检测
- 用户登录和注册系统
- 检测结果保存功能

## 安装说明

1. 克隆项目到本地：
```bash
git clone [你的仓库地址]
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 下载预训练模型：
- 项目已包含预训练模型文件：
  - yolo11n.pt（目标检测）
  - yolo11n-seg.pt（实例分割）
  - yolo11n-pose.pt（姿态估计）
  - yolo11n-cls.pt（图像分类）

## 使用说明

1. 运行程序：
```bash
python main.py
```

2. 登录系统：
   - 首次使用需要注册账号
   - 使用注册的账号密码登录

3. 主界面操作：
   - 选择检测任务类型
   - 选择输入源（图片/视频/摄像头）
   - 调整置信度阈值
   - 开始检测
   - 查看和保存结果

## 项目结构

```
├── main.py              # 程序入口
├── main_window.py       # 主窗口界面
├── login.py            # 登录界面
├── register.py         # 注册界面
├── user_manager.py     # 用户管理
├── yolo_detector.py    # YOLO检测核心
├── ui/                 # UI文件目录
├── icons/             # 图标资源
├── results/           # 检测结果保存目录
└── ultralytics/       # YOLO模型相关
```

## 功能展示

### 用户界面
#### 登录界面
![登录界面](docs/images/login.png)

#### 注册界面
![注册界面](docs/images/register.png)

### 检测功能展示
#### 目标检测
![目标检测示例](docs/images/detection.png)

#### 实例分割
![实例分割示例](docs/images/segmentation.png)

#### 姿态估计
![姿态估计示例](docs/images/pose.png)

#### 图像分类
![图像分类示例](docs/images/classification.png)

## 注意事项

- 确保系统已安装Python 3.8或更高版本
- 首次运行可能需要下载模型文件
- 建议使用GPU进行检测以获得更好的性能

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。
