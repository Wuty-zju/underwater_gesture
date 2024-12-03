import cv2
from tqdm import tqdm
from ultralytics import YOLO

# 加载预训练的YOLO模型
model_path = './runs/detect/train17/weights/best.pt'

# 视频文件输入输出路径
input_path = './examples/AUV_gesture.mov'
output_path = './examples/AUV_gesture_yolov8l.mp4'

# 加载预训练的YOLO模型
model = YOLO(model_path)

# 打开视频文件
cap = cv2.VideoCapture(input_path)

# 获取视频的宽度、高度和帧率
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 设置视频写入器
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# 获取视频总帧数
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 初始化FPS计算的变量
frame_count = 0
total_inference_time = 0.0

# 使用 tqdm 显示视频帧处理的进度
with tqdm(total=total_frames, desc="Processing", leave=True, dynamic_ncols=True) as pbar:
    # 读取视频帧
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model(frame)

            # 提取推理时间并计算实时帧率
            inference_time_ms = results[0].speed['inference']
            if inference_time_ms != 0:
                fps = 1000 / inference_time_ms
            total_inference_time += inference_time_ms / 1000
            frame_count += 1

            # 获取标注后的帧
            annotated_frame = results[0].plot()

            # 添加帧率信息到右上角
            fps_text = f'FPS {fps:.2f}'
            cv2.putText(annotated_frame, fps_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            # 写入标注帧到视频文件
            out.write(annotated_frame)

            # 更新进度条，打印到控制台
            pbar.update(1)
        else:
            break

# 计算并打印平均推理FPS
average_fps = frame_count / total_inference_time

# 释放视频读取和写入对象
cap.release()
out.release()

print('推理结果成功保存, 平均FPS: ', average_fps)
