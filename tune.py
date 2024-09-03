from ultralytics import YOLO

model = YOLO("yolov8n.yaml") # YOLOv8n/s/m/l/x or YOLOv9t/s/m/c/e YOLOv10n/s/m/b/l/x

model.tune(data="datasets/CADDY_gestures_YOLO/CADDY_gestures.yaml", epochs=30, iterations=300, optimizer="AdamW", plots=False, save=False, val=False)

'''
Start-Process -FilePath "C:/Users/wutia/Anaconda3/envs/uw_g/python.exe" -ArgumentList "C:/Users/wutia/Desktop/underwater_gesture/tune.py" -RedirectStandardOutput "C:/Users/wutia/Desktop/underwater_gesture/log/tune_1.log" -RedirectStandardError "C:/Users/wutia/Desktop/underwater_gesture/log/tune_2.log"
Start-Process -FilePath "C:/Users/wutia/Anaconda3/envs/uw_g/python.exe" -ArgumentList "C:/Users/wutia/Downloads/underwater_gesture/tune.py" -RedirectStandardOutput "C:/Users/wutia/Downloads/underwater_gesture/log/tune_1.log" -RedirectStandardError "C:/Users/wutia/Downloads/underwater_gesture/log/tune_2.log"

Get-Process -Name "python"
Get-Process -Name "python" | Stop-Process
'''