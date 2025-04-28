import cv2
import numpy as np
import torch
import os
from ultralytics import YOLO
import time
from pathlib import Path

# ------- 功能函数 -------

def load_model(model_type, model_path=None):
    """
    加载指定类型的YOLO模型
    
    参数:
        model_type: 模型类型，可选 'detection', 'segmentation', 'classification', 'pose'
        model_path: 模型路径，如果为None则使用默认模型
        
    返回:
        加载好的YOLO模型
    """
    if model_type == "detection":
        model_path = model_path if model_path else "yolo11n.pt"
    elif model_type == "segmentation":
        model_path = model_path if model_path else "yolo11n-seg.pt"
    elif model_type == "classification":
        model_path = model_path if model_path else "yolo11n-cls.pt"
    elif model_type == "pose":
        model_path = model_path if model_path else "yolo11n-pose.pt"
    else:
        raise ValueError("不支持的模型类型，请选择 'detection', 'segmentation', 'classification' 或 'pose'")
    
    print(f"正在加载{model_type}模型: {model_path}")
    model = YOLO(model_path)
    return model, model_type

def detect_image(model, model_type, image_path, conf=0.25, save_path=None, show=True):
    """
    处理单张图片
    
    参数:
        model: YOLO模型
        model_type: 模型类型
        image_path: 图片路径
        conf: 置信度阈值
        save_path: 保存结果的路径
        show: 是否显示结果
        
    返回:
        检测结果
    """
    print(f"正在处理图片: {image_path}")
    results = model.predict(source=image_path, conf=conf)
    
    # 显示结果
    if show:
        annotated_frame = results[0].plot()
        cv2.imshow("YOLO Detection", annotated_frame)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # 保存结果
    if save_path:
        save_results(results, model_type, save_path)
        
    return results

def detect_video(model, model_type, video_path, conf=0.25, save_path=None, show=True):
    """
    处理视频文件
    
    参数:
        model: YOLO模型
        model_type: 模型类型
        video_path: 视频路径
        conf: 置信度阈值
        save_path: 保存结果的路径
        show: 是否显示结果
        
    返回:
        检测结果
    """
    print(f"正在处理视频: {video_path}")
    
    # 处理视频并保存
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        results = model.predict(source=video_path, conf=conf, show=show, save=True, 
                              project=os.path.dirname(save_path), 
                              name=os.path.basename(save_path))
    else:
        results = model.predict(source=video_path, conf=conf, show=show)
        
    return results

def detect_webcam(model, model_type, camera_id=0, conf=0.25, save_path=None):
    """
    处理摄像头输入
    
    参数:
        model: YOLO模型
        model_type: 模型类型
        camera_id: 摄像头ID
        conf: 置信度阈值
        save_path: 保存结果的路径
    """
    print(f"正在启动摄像头 ID: {camera_id}")
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise ValueError(f"无法打开摄像头 ID: {camera_id}")
    
    # 获取视频宽度和高度
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:  # 某些摄像头可能会返回0的fps
        fps = 30
    
    # 如果需要保存视频
    out = None
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    
    print("按 'q' 键退出")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 使用模型预测
        start_time = time.time()
        
        if model_type == "pose":
            results = model.track(source=frame, conf=conf)
        else:
            results = model.predict(source=frame, conf=conf)
            
        # 计算FPS
        fps_current = 1.0 / (time.time() - start_time)
        
        # 绘制结果
        annotated_frame = results[0].plot()

        # 创建可写副本
        annotated_frame = annotated_frame.copy()  # 添加这一行
        
        # 在图像上显示FPS
        cv2.putText(annotated_frame, f"FPS: {int(fps_current)}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 显示结果
        cv2.imshow("YOLO Detection", annotated_frame)
        
        # 保存视频帧
        if out:
            out.write(annotated_frame)
            
        # 按q键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

def save_results(results, model_type, save_path):
    """
    保存检测结果到文本文件
    
    参数:
        results: 检测结果
        model_type: 模型类型
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        for i, result in enumerate(results):
            f.write(f"Result {i}:\n")
            
            if model_type == "detection" or model_type == "segmentation" or model_type == "pose":
                # 保存边界框信息
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    f.write(f"Boxes:\n")
                    for j, box in enumerate(result.boxes):
                        cls_id = int(box.cls[0]) if isinstance(box.cls[0], torch.Tensor) else int(box.cls[0])
                        cls_name = result.names[cls_id]
                        conf = float(box.conf[0]) if isinstance(box.conf[0], torch.Tensor) else float(box.conf[0])
                        xyxy = [float(x) for x in box.xyxy[0]]
                        f.write(f"  Box {j}: Class={cls_name}, Confidence={conf:.4f}, Coordinates={xyxy}\n")
                
                # 保存分割信息
                if hasattr(result, 'masks') and result.masks is not None:
                    f.write(f"Segmentation masks: {len(result.masks)} found\n")
                
                # 保存姿态关键点信息
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    f.write(f"Pose keypoints: {len(result.keypoints)} instances found\n")
                    
            elif model_type == "classification":
                # 保存分类信息
                if hasattr(result, 'probs'):
                    # 处理分类结果
                    try:
                        # 如果是PyTorch张量
                        if hasattr(result.probs, 'top5') and hasattr(result.probs, 'top5conf'):
                            top_indices = result.probs.top5
                            top_probs = result.probs.top5conf
                            
                            f.write(f"Classification results:\n")
                            for idx, prob in zip(top_indices, top_probs):
                                idx_val = idx.item() if hasattr(idx, 'item') else idx
                                prob_val = prob.item() if hasattr(prob, 'item') else prob
                                f.write(f"  Class: {result.names[idx_val]}, Probability: {prob_val:.4f}\n")
                    except (AttributeError, TypeError) as e:
                        f.write(f"Classification results available but format error: {str(e)}\n")
            
            f.write("\n")
    
    print(f"检测结果已保存到: {save_path}")

def get_file_type(file_path):
    """
    判断文件类型
    
    参数:
        file_path: 文件路径
        
    返回:
        文件类型 'image', 'video' 或 None
    """
    if not file_path:
        return None
    
    # 检查是否为数字（摄像头ID）
    if str(file_path).isdigit():
        return "webcam"
    
    # 检查文件扩展名
    ext = str(file_path).lower().split('.')[-1]
    
    if ext in ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff']:
        return "image"
    elif ext in ['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv']:
        return "video"
    
    return None

# ------- 测试模块 -------

def test_detection():
    """测试目标检测功能"""
    print("\n=== 测试目标检测功能 ===")
    
    # 加载模型
    model, model_type = load_model("detection")
    
    # 测试图片
    image_path = input("请输入测试图片路径 (按回车跳过): ")
    if image_path:
        results = detect_image(model, model_type, image_path)
        
        # 打印结果信息
        for r in results:
            print(f"检测到 {len(r.boxes)} 个目标")
            for i, box in enumerate(r.boxes):
                cls_id = int(box.cls[0]) if isinstance(box.cls[0], torch.Tensor) else int(box.cls[0])
                cls_name = r.names[cls_id]
                conf = float(box.conf[0]) if isinstance(box.conf[0], torch.Tensor) else float(box.conf[0])
                print(f"  目标 {i+1}: {cls_name}, 置信度: {conf:.4f}")
    
    # 测试视频
    video_path = input("\n请输入测试视频路径 (按回车跳过): ")
    if video_path:
        detect_video(model, model_type, video_path)
    
    # 测试摄像头
    test_webcam = input("\n是否测试摄像头? (y/n): ")
    if test_webcam.lower() == 'y':
        camera_id = int(input("请输入摄像头ID (默认0): ") or "0")
        detect_webcam(model, model_type, camera_id)

def test_segmentation():
    """测试图像分割功能"""
    print("\n=== 测试图像分割功能 ===")
    
    # 加载模型
    model, model_type = load_model("segmentation")
    
    # 测试图片
    image_path = input("请输入测试图片路径 (按回车跳过): ")
    if image_path:
        results = detect_image(model, model_type, image_path)
        
        # 打印结果信息
        for r in results:
            if hasattr(r, 'masks') and r.masks is not None:
                print(f"检测到 {len(r.masks)} 个分割区域")
            else:
                print("未检测到分割区域")
    
    # 测试视频
    video_path = input("\n请输入测试视频路径 (按回车跳过): ")
    if video_path:
        detect_video(model, model_type, video_path)
    
    # 测试摄像头
    test_webcam = input("\n是否测试摄像头? (y/n): ")
    if test_webcam.lower() == 'y':
        camera_id = int(input("请输入摄像头ID (默认0): ") or "0")
        detect_webcam(model, model_type, camera_id)

def test_classification():
    """测试图像分类功能"""
    print("\n=== 测试图像分类功能 ===")
    
    # 加载模型
    model, model_type = load_model("classification")
    
    # 测试图片
    image_path = input("请输入测试图片路径 (按回车跳过): ")
    if image_path:
        results = detect_image(model, model_type, image_path)
        
        # 打印结果信息
        for r in results:
            if hasattr(r, 'probs'):
                try:
                    # 如果是PyTorch张量
                    if hasattr(r.probs, 'top5') and hasattr(r.probs, 'top5conf'):
                        top_indices = r.probs.top5
                        top_probs = r.probs.top5conf
                        
                        print("分类结果 (Top 5):")
                        for idx, prob in zip(top_indices, top_probs):
                            idx_val = idx.item() if hasattr(idx, 'item') else idx
                            prob_val = prob.item() if hasattr(prob, 'item') else prob
                            print(f"  {r.names[idx_val]}: {prob_val:.4f}")
                except (AttributeError, TypeError) as e:
                    print(f"无法解析分类结果: {str(e)}")
                    print("结果格式:", type(r.probs))
    
    # 测试视频
    video_path = input("\n请输入测试视频路径 (按回车跳过): ")
    if video_path:
        detect_video(model, model_type, video_path)
    
    # 测试摄像头
    test_webcam = input("\n是否测试摄像头? (y/n): ")
    if test_webcam.lower() == 'y':
        camera_id = int(input("请输入摄像头ID (默认0): ") or "0")
        detect_webcam(model, model_type, camera_id)

def test_pose():
    """测试姿态估计功能"""
    print("\n=== 测试姿态估计功能 ===")
    
    # 加载模型
    model, model_type = load_model("pose")
    
    # 测试图片
    image_path = input("请输入测试图片路径 (按回车跳过): ")
    if image_path:
        results = detect_image(model, model_type, image_path)
        
        # 打印结果信息
        for r in results:
            if hasattr(r, 'keypoints') and r.keypoints is not None:
                print(f"检测到 {len(r.keypoints)} 个人体姿态")
            else:
                print("未检测到人体姿态")
    
    # 测试视频
    video_path = input("\n请输入测试视频路径 (按回车跳过): ")
    if video_path:
        detect_video(model, model_type, video_path)
    
    # 测试摄像头
    test_webcam = input("\n是否测试摄像头? (y/n): ")
    if test_webcam.lower() == 'y':
        camera_id = int(input("请输入摄像头ID (默认0): ") or "0")
        detect_webcam(model, model_type, camera_id)

def custom_test():
    """自定义测试功能"""
    print("\n=== 自定义测试 ===")
    
    # 选择模型类型
    print("请选择模型类型:")
    print("1. 目标检测 (detection)")
    print("2. 图像分割 (segmentation)")
    print("3. 图像分类 (classification)")
    print("4. 姿态估计 (pose)")
    
    model_choice = input("请输入选项 (1-4): ")
    model_types = {
        "1": "detection",
        "2": "segmentation",
        "3": "classification",
        "4": "pose"
    }
    
    if model_choice not in model_types:
        print("无效选项")
        return
    
    model_type = model_types[model_choice]
    
    # 指定模型路径
    model_path = input("请输入模型路径 (按回车使用默认模型): ")
    model_path = model_path if model_path else None
    
    # 加载模型
    model, model_type = load_model(model_type, model_path)
    
    # 选择输入源
    print("\n请选择输入源类型:")
    print("1. 图片")
    print("2. 视频")
    print("3. 摄像头")
    
    source_choice = input("请输入选项 (1-3): ")
    
    # 设置置信度阈值
    conf = float(input("\n请输入置信度阈值 (0-1, 默认0.25): ") or "0.25")
    
    # 设置保存路径
    save_path = input("\n请输入保存结果的路径 (按回车不保存): ")
    save_path = save_path if save_path else None
    
    # 根据输入源类型处理
    if source_choice == "1":  # 图片
        image_path = input("请输入图片路径: ")
        detect_image(model, model_type, image_path, conf, save_path)
    elif source_choice == "2":  # 视频
        video_path = input("请输入视频路径: ")
        detect_video(model, model_type, video_path, conf, save_path)
    elif source_choice == "3":  # 摄像头
        camera_id = int(input("请输入摄像头ID (默认0): ") or "0")
        detect_webcam(model, model_type, camera_id, conf, save_path)
    else:
        print("无效选项")

def run_tests():
    """运行所有功能测试"""
    print("=== YOLO 多功能检测器测试 ===")
    
    while True:
        print("\n请选择要测试的功能:")
        print("1. 目标检测")
        print("2. 图像分割")
        print("3. 图像分类")
        print("4. 姿态估计")
        print("5. 自定义测试")
        print("0. 退出")
        
        choice = input("\n请输入选项 (0-5): ")
        
        if choice == "1":
            test_detection()
        elif choice == "2":
            test_segmentation()
        elif choice == "3":
            test_classification()
        elif choice == "4":
            test_pose()
        elif choice == "5":
            custom_test()
        elif choice == "0":
            print("测试结束")
            break
        else:
            print("无效选项，请重新输入")

# 主函数
if __name__ == "__main__":
    run_tests()

class YOLODetector:
    def __init__(self, model_path):
        """初始化YOLO检测器"""
        self.model_path = model_path
        self.model = None
        self.model_type = None  # 用于存储模型类型
        self.confidence = 0.5
        self.iou = 0.5
        self.line_width = 3
        
        # 加载模型
        self.load_model()
        
    def load_model(self):
        """加载YOLO模型"""
        try:
            self.model = YOLO(self.model_path)
            # 根据模型文件名判断模型类型
            if "seg" in self.model_path.lower():
                self.model_type = "segmentation"
            elif "cls" in self.model_path.lower():
                self.model_type = "classification"
            elif "pose" in self.model_path.lower():
                self.model_type = "pose"
            else:
                self.model_type = "detection"
            print(f"模型 {self.model_path} 加载成功，类型: {self.model_type}")
        except Exception as e:
            error_msg = f"加载模型 {self.model_path} 失败: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)
            
    def detect(self, image):
        """执行目标检测"""
        if self.model is None:
            raise Exception("模型未加载")
            
        # 执行检测
        results = self.model(image, conf=self.confidence, iou=self.iou)
        
        # 获取检测结果
        result_image = image.copy()
        detections = []
        
        for result in results:
            if self.model_type == "detection":
                # 目标检测模式
                boxes = result.boxes
                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # 获取类别和置信度
                    cls = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    
                    # 获取类别名称
                    cls_name = self.model.names[cls]
                    
                    # 添加到检测结果
                    detections.append({
                        "class": cls_name,
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2]
                    })
                    
            elif self.model_type == "segmentation":
                # 分割模式
                if hasattr(result, 'masks') and result.masks is not None:
                    masks = result.masks.data
                    for mask in masks:
                        # 将掩码转换为二值图像
                        mask_np = mask.cpu().numpy()
                        mask_np = (mask_np * 255).astype(np.uint8)
                        
                        # 获取掩码的轮廓
                        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for contour in contours:
                            # 获取轮廓的边界框
                            x, y, w, h = cv2.boundingRect(contour)
                            detections.append({
                                "type": "segmentation",
                                "bbox": [x, y, x+w, y+h]
                            })
                            
            elif self.model_type == "classification":
                # 分类模式
                if hasattr(result, 'probs'):
                    probs = result.probs
                    if hasattr(probs, 'top5') and hasattr(probs, 'top5conf'):
                        top_indices = probs.top5
                        top_probs = probs.top5conf
                        
                        for idx, prob in zip(top_indices, top_probs):
                            idx_val = idx.item() if hasattr(idx, 'item') else idx
                            prob_val = prob.item() if hasattr(prob, 'item') else prob
                            cls_name = self.model.names[idx_val]
                            
                            detections.append({
                                "type": "classification",
                                "class": cls_name,
                                "confidence": prob_val
                            })
                            
            elif self.model_type == "pose":
                # 姿态估计模式
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    keypoints = result.keypoints.data
                    for kpts in keypoints:
                        detections.append({
                            "type": "pose",
                            "keypoints": kpts.cpu().numpy().tolist()
                        })
        
        # 使用 YOLO 的 plot 方法绘制结果，传入线宽参数
        result_image = results[0].plot(line_width=self.line_width)
                
        return result_image, detections
        
    @staticmethod
    def get_available_cameras():
        """检测系统中可用的摄像头
        
        Returns:
            list: 可用的摄像头ID列表
        """
        available_cameras = []
        # 尝试打开前10个摄像头
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        return available_cameras
        
    def detect_webcam(self, camera_id=None, save_path=None, show=True):
        """执行摄像头检测
        
        Args:
            camera_id: 摄像头ID，如果为None则自动选择第一个可用的摄像头
            save_path: 保存结果视频的路径，如果为None则不保存
            show: 是否显示检测结果
        """
        if self.model is None:
            raise Exception("模型未加载")
            
        # 如果没有指定摄像头ID，则获取可用的摄像头列表
        if camera_id is None:
            available_cameras = self.get_available_cameras()
            if not available_cameras:
                raise Exception("未找到可用的摄像头")
            camera_id = available_cameras[0]
            print(f"使用摄像头 ID: {camera_id}")
            
        # 打开摄像头
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise Exception(f"无法打开摄像头 ID: {camera_id}")
            
        # 获取摄像头信息
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:  # 某些摄像头可能会返回0的fps
            fps = 30
            
        # 创建视频写入器
        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
            
        # 处理视频帧
        frame_count = 0
        start_time = time.time()
        
        print("按 'q' 键退出")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # 执行检测
            result_frame, _ = self.detect(frame)
            
            # 显示FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps_current = frame_count / elapsed_time
            cv2.putText(result_frame, f"FPS: {fps_current:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示结果
            if show:
                cv2.imshow("YOLO Detection", result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            # 保存结果
            if writer:
                writer.write(result_frame)
                
        # 释放资源
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()
            
        print("摄像头检测结束")