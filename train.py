from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO('yolov8n.yaml') # YOLOv8n/s/m/l/x or YOLOv9t/s/m/c/e YOLOv10n/s/m/b/l/x
    #model = YOLO('yolov9s.pt')

    model.info()

    results = model.train(data='datasets/CADDY_gestures_complete_YOLO/CADDY_gestures_complete.yaml', 
                          epochs=100, 
                          batch=4, 
                          patience=0, 
                          pretrained=False)

'''
# wty_lab_pc
Start-Process -FilePath python.exe -ArgumentList train.py -RedirectStandardOutput log/train_1.log -RedirectStandardError log/train_2.log
Start-Process -FilePath "C:/Users/wutia/Anaconda3/envs/uw_g/python.exe" -ArgumentList "C:/Users/wutia/Desktop/underwater_gesture/train.py" -RedirectStandardOutput "C:/Users/wutia/Desktop/underwater_gesture/log/train_1.log" -RedirectStandardError "C:/Users/wutia/Desktop/underwater_gesture/log/train_2.log"
# windows
Start-Process -FilePath "C:/Users/wutia/Anaconda3/envs/uw_g/python.exe" -ArgumentList "C:/Users/wutia/Downloads/underwater_gesture/train.py" -RedirectStandardOutput "C:/Users/wutia/Downloads/underwater_gesture/log/train_1.log" -RedirectStandardError "C:/Users/wutia/Downloads/underwater_gesture/log/train_2.log"

# 读取和打印日志
Get-Content -Path "C:/Users/wutia/Desktop/underwater_gesture/log/train_2.log" -Wait

# 查找和停止进程
Get-Process -Name "python"
Get-Process -Name "python" | Stop-Process

# 性能监控
while ($true) { $cpu = [math]::round((Get-WmiObject Win32_Processor | Measure-Object LoadPercentage -Average).Average, 2); $totalMemory = (Get-WmiObject Win32_ComputerSystem).TotalPhysicalMemory; $freeMemory = (Get-WmiObject Win32_OperatingSystem).FreePhysicalMemory * 1KB; $usedMemory = $totalMemory - $freeMemory; $memoryUsagePercent = [math]::round($usedMemory / $totalMemory * 100, 2); $gpuUtil, $gpuMemUtil, $gpuMemTotal, $gpuMemUsed = (& nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.used --format=csv,noheader,nounits).Split(",").Trim(); $gpuUtil = [math]::round([double]$gpuUtil, 2); $gpuMemUsagePercent = [math]::round($gpuMemUsed / $gpuMemTotal * 100, 2); Clear-Host; Write-Host "CPU Usage: $cpu%"; Write-Host "Memory Usage: $([math]::round($usedMemory / 1MB, 2)) MB / $([math]::round($totalMemory / 1MB, 2)) MB ($memoryUsagePercent%)"; Write-Host "GPU Usage: $gpuUtil%"; Write-Host "GPU Memory Usage: $gpuMemUsed MB / $gpuMemTotal MB ($gpuMemUsagePercent%)"; Start-Sleep 1 }
'''
