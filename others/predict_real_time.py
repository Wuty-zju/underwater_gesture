# 如果无法还原水下的真实环境，建议使用 predict_video.py 
# 进行测试，该脚本用于实时检测摄像头捕获的视频帧
import cv2
from tqdm import tqdm
from ultralytics import YOLO

# 加载预训练的YOLO模型
model_path = './runs/detect/train17/weights/best.pt'
model = YOLO(model_path)

# 打开摄像头
cap = cv2.VideoCapture(0)

# 获取摄像头的宽度、高度和帧率
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 设置视频写入器
output_path = './examples/real_time.mp4'
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# 初始化FPS计算的变量
frame_count = 0
total_inference_time = 0.0

# 使用 tqdm 显示视频帧处理的进度
try:
    while True:
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

            # 显示处理后的帧
            cv2.imshow('YOLOv8 Detection', annotated_frame)

            # 按下 'q' 键退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
except KeyboardInterrupt:
    print("Processing interrupted")

# 计算并打印平均推理FPS
average_fps = frame_count / total_inference_time

# 释放视频读取和写入对象
cap.release()
out.release()
cv2.destroyAllWindows()

print('推理结果成功保存, 平均FPS: ', average_fps)