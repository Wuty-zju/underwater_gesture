from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO("yolov10n.yaml") # YOLOv8n/s/m/l/x or YOLOv9t/s/m/c/e YOLOv10n/s/m/b/l/x

    model.tune(
        data="datasets/CADDY_gestures_YOLO/CADDY_gestures.yaml",
        epochs=50,
        iterations=300
        )

'''
# 挂起进程
Start-Process -FilePath python.exe -ArgumentList tune_yolo.py -RedirectStandardOutput log/tune_yolo_1.log -RedirectStandardError log/tune_yolo_2.log

# 查找和停止进程
Get-Process -Name "python"
Get-Process -Name "python" | Stop-Process

# 启动进程并将输出重定向到日志文件
nohup python3 tune_yolo.py > log/tune_yolo_1.log 2> log/tune_yolo_2.log &

# 查找和停止进程
ps -C python3
pkill python3


# 性能监控
while ($true) { $cpu = [math]::round((Get-WmiObject Win32_Processor | Measure-Object LoadPercentage -Average).Average, 2); $totalMemory = (Get-WmiObject Win32_ComputerSystem).TotalPhysicalMemory; $freeMemory = (Get-WmiObject Win32_OperatingSystem).FreePhysicalMemory * 1KB; $usedMemory = $totalMemory - $freeMemory; $memoryUsagePercent = [math]::round($usedMemory / $totalMemory * 100, 2); $gpuUtil, $gpuMemUtil, $gpuMemTotal, $gpuMemUsed = (& nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.used --format=csv,noheader,nounits).Split(",").Trim(); $gpuUtil = [math]::round([double]$gpuUtil, 2); $gpuMemUsagePercent = [math]::round($gpuMemUsed / $gpuMemTotal * 100, 2); Clear-Host; Write-Host "CPU Usage: $cpu%"; Write-Host "Memory Usage: $([math]::round($usedMemory / 1MB, 2)) MB / $([math]::round($totalMemory / 1MB, 2)) MB ($memoryUsagePercent%)"; Write-Host "GPU Usage: $gpuUtil%"; Write-Host "GPU Memory Usage: $gpuMemUsed MB / $gpuMemTotal MB ($gpuMemUsagePercent%)"; Start-Sleep 1 }
'''