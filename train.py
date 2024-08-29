from ultralytics import YOLO

if __name__ == '__main__':

    # Build a YOLO model from scratch
    model = YOLO('yolov8m.yaml') # YOLOv8n/s/m/l/x or YOLOv9t/s/m/c/e YOLOv10n/s/m/b/l/x

    # Build a YOLOv9c model from pretrained weight
    #model = YOLO('yolov9s.pt') # YOLOv8n/s/m/l or YOLOv9c

    # Display model information (optional)
    model.info()

    # Train the model on the COCO8 example dataset for 100 epochs
    results = model.train(data='datasets/CADDY_gestures_YOLO/CADDY_gestures.yaml', epochs=100, batch=4, patience=0)
'''
Start-Job -ScriptBlock {&'C:/Users/wutia/Anaconda3/envs/uw_g/python.exe' train.py > train.log 2>&1}
Start-Job -ScriptBlock {& 'C:/Users/wutia/Anaconda3/envs/uw_g/python.exe' 'C:/Users/wutia/Desktop/underwater_gesture/train.py' > 'C:/Users/wutia/Desktop/underwater_gesture/train.log' 2>&1}
Start-Process -FilePath "C:/Users/wutia/Anaconda3/envs/uw_g/python.exe" -ArgumentList "C:/Users/wutia/Desktop/underwater_gesture/train.py" -RedirectStandardOutput "C:/Users/wutia/Desktop/underwater_gesture/log/train_1.log" -RedirectStandardError "C:/Users/wutia/Desktop/underwater_gesture/log/train_2.log"
Start-Process -FilePath "C:/Users/wutia/Anaconda3/envs/uw_g/python.exe" -ArgumentList "C:/Users/wutia/Downloads/underwater_gesture/train.py" -RedirectStandardOutput "C:/Users/wutia/Downloads/underwater_gesture/log/train_1.log" -RedirectStandardError "C:/Users/wutia/Downloads/underwater_gesture/log/train_2.log"
$process1 = Start-Process -FilePath "C:/Users/wutia/Anaconda3/envs/uw_g/python.exe" -ArgumentList "C:/Users/wutia/Desktop/underwater_gesture/train.py" -RedirectStandardOutput "C:/Users/wutia/Desktop/underwater_gesture/log/train_1.log" -RedirectStandardError "C:/Users/wutia/Desktop/underwater_gesture/log/train_2.log" -PassThru; $process1 | Wait-Process; Start-Process -FilePath "C:/Users/wutia/Anaconda3/envs/uw_g/python.exe" -ArgumentList "C:/Users/wutia/Desktop/underwater_gesture/train_2.py" -RedirectStandardOutput "C:/Users/wutia/Desktop/underwater_gesture/log/train_3.log" -RedirectStandardError "C:/Users/wutia/Desktop/underwater_gesture/log/train_4.log"

Get-Content -Path "C:/Users/wutia/Desktop/underwater_gesture/log/train_2.log" -Wait
Get-Process -Name "python"
Get-Process -Name "python" | Stop-Process

while ($true) { $cpu = Get-WmiObject -Class Win32_Processor | Measure-Object -Property LoadPercentage -Average | Select-Object -ExpandProperty Average; $totalMemory = (Get-WmiObject -Class Win32_ComputerSystem).TotalPhysicalMemory; $freeMemory = (Get-WmiObject -Class Win32_OperatingSystem).FreePhysicalMemory * 1KB; $usedMemory = $totalMemory - $freeMemory; $memoryUsagePercent = [math]::round(($usedMemory / $totalMemory) * 100, 2); $gpuInfo = & nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.used --format=csv,noheader,nounits; $gpuUtil = $gpuInfo.Split(",")[0].Trim(); $gpuMemUtil = $gpuInfo.Split(",")[1].Trim(); $gpuMemTotal = $gpuInfo.Split(",")[2].Trim(); $gpuMemUsed = $gpuInfo.Split(",")[3].Trim(); Clear-Host; Write-Host "CPU Usage: $cpu%"; Write-Host "Memory Usage: $([math]::round($usedMemory / 1MB, 2)) MB ($memoryUsagePercent%)"; Write-Host "GPU Usage: $gpuUtil%"; Write-Host "GPU Memory Usage: $gpuMemUsed MB / $gpuMemTotal MB ($gpuMemUtil%)"; Start-Sleep -Seconds 1 }
'''