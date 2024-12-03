# predict_video.py 的可视化交互前端版本
import gradio as gr
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from pathlib import Path

def predict(model_path, input_video):
    # Create a directory for outputs if it doesn't exist
    output_dir = Path("./examples")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "output_video.mp4"

    model = YOLO(model_path)
    cap = cv2.VideoCapture(input_video.name)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    total_inference_time = 0.0
    
    with tqdm(total=total_frames, desc="Processing", leave=True, dynamic_ncols=True) as pbar:
        while cap.isOpened():
            success, frame = cap.read()
            if success:
                results = model(frame)
                inference_time_ms = results[0].speed['inference']
                if inference_time_ms != 0:
                    fps = 1000 / inference_time_ms
                total_inference_time += inference_time_ms / 1000
                frame_count += 1

                annotated_frame = results[0].plot()
                fps_text = f'FPS {fps:.2f}'
                cv2.putText(annotated_frame, fps_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                out.write(annotated_frame)
                pbar.update(1)
            else:
                break

    average_fps = frame_count / total_inference_time
    cap.release()
    out.release()
    
    return str(output_path), f'推理结果成功保存, 平均FPS: {average_fps:.2f}'

iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Model Path", value="./runs/detect/train12/weights/best.pt", type="text"),
        gr.File(label="Upload Video", type="filepath")
    ],
    outputs=[
        gr.File(label="Download Processed Video"),
        gr.Text(label="Result")
    ],
    title="YOLO Video Processing",
    description="Upload a video and choose a model to process the video with YOLO."
)

iface.launch(share=True)