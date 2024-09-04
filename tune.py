from ultralytics import YOLO
from datetime import datetime

if __name__ == '__main__':

    #wandb.login(key="e10766eb91aad0e2c30a2ecb8020f5be6eff9211")

    model = YOLO("yolov10n.yaml") # YOLOv8n/s/m/l/x or YOLOv9t/s/m/c/e YOLOv10n/s/m/b/l/x

    current_date = datetime.now().strftime("%Y%m%d")

    trial_dirname = f"{model}_tune_{current_date[:8]}"  # 只使用前8位的日期

    result_grid = model.tune(
        data="datasets/CADDY_gestures_YOLO/CADDY_gestures.yaml",
        epochs=50,
        iterations=300,
        use_ray=True,
        trial_dirname_creator=trial_dirname
    )
'''
# 挂起进程
Start-Process -FilePath python.exe -ArgumentList tune.py -RedirectStandardOutput log/tune_1.log -RedirectStandardError log/tune_2.log

# 查找和停止进程
Get-Process -Name "python"
Get-Process -Name "python" | Stop-Process

# 性能监控
while ($true) { $cpu = [math]::round((Get-WmiObject Win32_Processor | Measure-Object LoadPercentage -Average).Average, 2); $totalMemory = (Get-WmiObject Win32_ComputerSystem).TotalPhysicalMemory; $freeMemory = (Get-WmiObject Win32_OperatingSystem).FreePhysicalMemory * 1KB; $usedMemory = $totalMemory - $freeMemory; $memoryUsagePercent = [math]::round($usedMemory / $totalMemory * 100, 2); $gpuUtil, $gpuMemUtil, $gpuMemTotal, $gpuMemUsed = (& nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.used --format=csv,noheader,nounits).Split(",").Trim(); $gpuUtil = [math]::round([double]$gpuUtil, 2); $gpuMemUsagePercent = [math]::round($gpuMemUsed / $gpuMemTotal * 100, 2); Clear-Host; Write-Host "CPU Usage: $cpu%"; Write-Host "Memory Usage: $([math]::round($usedMemory / 1MB, 2)) MB / $([math]::round($totalMemory / 1MB, 2)) MB ($memoryUsagePercent%)"; Write-Host "GPU Usage: $gpuUtil%"; Write-Host "GPU Memory Usage: $gpuMemUsed MB / $gpuMemTotal MB ($gpuMemUsagePercent%)"; Start-Sleep 1 }
'''