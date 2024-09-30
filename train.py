from torch import device
from ultralytics import YOLO


model_configs = ["yolo11n.yaml", "yolo11s.yaml", "yolo11m.yaml", "yolo11l.yaml", "yolo11x.yaml"]


def train_yolo_model(model_config):
    model = YOLO(model_config)
    results = model.train(
        data="datasets/CADDY_gestures_complete_YOLO/CADDY_gestures_complete.yaml",
        epochs=1000,
        batch=4,
        patience=0,
        pretrained=False,
        device=[0],
    )

# 主程序
if __name__ == '__main__':
    for config in model_configs:
        train_yolo_model(config)

'''
# 挂起进程
$timestamp = (Get-Date).ToString("yyyyMMdd_HHmmss"); $p = Start-Process -FilePath python.exe -ArgumentList "train.py" -RedirectStandardOutput "log/train_${timestamp}_1.log" -RedirectStandardError "log/train_${timestamp}_2.log"

# 查找和停止进程
Get-Process -Name "python"
Get-Process -Name "python" | Stop-Process

# 挂起进程
nohup bash -c 'python train.py & PID=$!; echo "PID: $PID"; wait $PID' &> "log/train_$(date +%Y%m%d_%H%M%S).log" &

# 性能监控
while ($true) { $cpu = [math]::round((Get-WmiObject Win32_Processor | Measure-Object LoadPercentage -Average).Average, 2); $totalMemory = (Get-WmiObject Win32_ComputerSystem).TotalPhysicalMemory; $freeMemory = (Get-WmiObject Win32_OperatingSystem).FreePhysicalMemory * 1KB; $usedMemory = $totalMemory - $freeMemory; $memoryUsagePercent = [math]::round($usedMemory / $totalMemory * 100, 2); $gpuUtil, $gpuMemUtil, $gpuMemTotal, $gpuMemUsed = (& nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.used --format=csv,noheader,nounits).Split(",").Trim(); $gpuUtil = [math]::round([double]$gpuUtil, 2); $gpuMemUsagePercent = [math]::round($gpuMemUsed / $gpuMemTotal * 100, 2); Clear-Host; Write-Host "CPU Usage: $cpu%"; Write-Host "Memory Usage: $([math]::round($usedMemory / 1MB, 2)) MB / $([math]::round($totalMemory / 1MB, 2)) MB ($memoryUsagePercent%)"; Write-Host "GPU Usage: $gpuUtil%"; Write-Host "GPU Memory Usage: $gpuMemUsed MB / $gpuMemTotal MB ($gpuMemUsagePercent%)"; Start-Sleep 1 }
'''
