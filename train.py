from ultralytics import YOLO
from datetime import datetime
import json
import os

model_configs = [
    #"yolov8n.yaml", "yolov8s.yaml", "yolov8m.yaml", "yolov8l.yaml", "yolov8x.yaml",           # YOLOv8 模型
    #"yolov9t.yaml", "yolov9s.yaml", "yolov9m.yaml", "yolov9c.yaml", "yolov9e.yaml",           # YOLOv9 模型
    #"yolov10n.yaml", "yolov10s.yaml", "yolov10m.yaml", "yolov10l.yaml", "yolov10x.yaml",      # YOLOv10 模型
    #"yolo11n.yaml", "yolo11s.yaml", "yolo11m.yaml", "yolo11l.yaml", "yolo11x.yaml"            # YOLO11 模型
    
    #"yolo11-C3k2-AdditiveBlock.yaml",          # sample跑完
    #"yolo11-C3k2-SCConv.yaml",         # sample跑完
    #"yolo11-C3k2-KAN.yaml",            # annot access local variable 'kan' where it is not associated with a value(m/l/x)
    #"yolo11-ReCalibrationFPN-P345.yaml",         # sample跑完
    #"yolo11-WaveletPool.yaml",         # sample跑完
    #"yolo11-efficientViT.yaml",            # sample跑完（缺n/s）
    #"yolo11-C3k2-WTConv.yaml",         # sample跑完
    #"yolo11-LADH.yaml",                # sample跑完
    #"yolo11-RSCD.yaml",                # sample跑完
    #"yolo11-C3k2-EIEM.yaml",           # sample跑完（缺l/x）
    #"yolo11-ReCalibrationFPN-P2345.yaml",                # sample跑完
    #"yolo11-FeaturePyramidSharedConv.yaml",                # sample跑完
    #"yolo11-C3k2-LFE.yaml",                # sample跑完
    #"yolo11-C2BRA.yaml",                # sample跑完
    #"yolo11-C2CGA.yaml",                # sample跑完
    #"yolo11-C3k2-MutilScaleEdgeInformationSelect.yaml",                 # sample跑完
    #"yolo11-C3k2-RAB.yaml",           # sample跑完 nan（关闭amp，x模型val出错）
    #"yolo11x-C3k2-HDRAB.yaml", "yolo11l-C3k2-HDRAB.yaml", "yolo11m-C3k2-HDRAB.yaml", "yolo11s-C3k2-HDRAB.yaml", "yolo11n-C3k2-HDRAB.yaml", # 尝试amp（nan,关闭amp，x模型val出错）
    
    #"yolo11n-C3k2-EMSC.yaml", "yolo11s-C3k2-EMSC.yaml", "yolo11x-C3k2-EMSC.yaml", "yolo11l-C3k2-EMSC.yaml", "yolo11m-C3k2-EMSC.yaml",
    #"yolo11n-C3k2-EMSCP.yaml", "yolo11s-C3k2-EMSCP.yaml", "yolo11m-C3k2-EMSCP.yaml", "yolo11l-C3k2-EMSCP.yaml", "yolo11x-C3k2-EMSCP.yaml",
    #"yolo11n-C2DPB.yaml", "yolo11s-C2DPB.yaml", "yolo11m-C2DPB.yaml", "yolo11l-C2DPB.yaml", "yolo11x-C2DPB.yaml", 
    
    #"yolo11-C3k2-SFA.yaml", "yolo11-C3k2-SFA.yaml", "yolo11-C3k2-SFA.yaml", "yolo11-C3k2-SFA.yaml", "yolo11-C3k2-SFA.yaml", 
    #"yolo11-C3k2-CTA.yaml", 
    
    #"yolo11-GlobalEdgeInformationTransfer1.yaml", 
    #"yolo11-GlobalEdgeInformationTransfer2.yaml", 
    #"yolo11-GlobalEdgeInformationTransfer3.yaml", 
    
    #"yolo11-C3k2-IDWC.yaml", 
    #"yolo11-C3k2-IDWD.yaml", 
    #"yolo11-inceptionnext.yaml", 
    
    #"yolo11-ReCalibrationFPN-P3456.yaml",
    #"yolo11-C3k2-MSBlock.yaml",
    #"yolo11-ContextGuideFPN.yaml",
    #"yolo11-C3k2-DLKA.yaml", 
    #"yolo11-FDPN.yaml",
    #"yolo11-FDPN-TADDH.yaml",
    #"yolo11-FDPN-DASI.yaml",
    #"yolo11-ASF-P2.yaml", 
    #"yolo11-C3k2-ContextGuided.yaml", 
    #"yolo11-C3k2-DCNV2-Dynamic.yaml",
    #"yolo11-EIEStem.yaml",
    #"yolo11-goldyolo-asf.yaml",
    #"yolo11-AIFI.yaml", 
    #"yolo11-C3k2-RFAConv.yaml", 
    #"yolo11-nmsfree.yaml",
]

datasets = {
    "sample": "datasets/CADDY_gestures_sample/yolo/data.yaml",
    #"complete": "datasets/CADDY_gestures_complete/yolo/data.yaml"
}

hyperparameters = {
    "device": [0],             # 设备: 单GPU ("0"), 多GPU ("0,1"), 或 CPU ("cpu")
    "amp": True,               # 是否启用混合精度训练(默认启用)
    "epochs": 500,             # 训练轮数
    "batch_size": 8,           # 每批次图像数量
    "imgsz": 640,              # 输入图像尺寸
    "patience": 0,             # 提前停止的耐心值
    "pretrained": False,       # 是否使用预训练权重
    "save": True,              # 是否保存训练结果
    "plots": True              # 是否生成训练曲线
}

def train_model(train_model_config, data_path, hyperparameters, train_save_dir, name="train"):
    """
    训练 YOLO 模型。

    参数:
        train_model_config (str): 模型配置文件路径。
        data_path (str): 数据集配置文件路径。
        hyperparameters (dict): 超参数配置字典。
        save_dir (str): 训练结果保存目录。
    """
    model = YOLO(train_model_config)
    
    model.train(
        data=data_path,
        epochs=hyperparameters["epochs"],
        batch=hyperparameters["batch_size"],
        imgsz=hyperparameters["imgsz"],
        patience=hyperparameters["patience"],
        pretrained=hyperparameters["pretrained"],
        save=hyperparameters["save"],
        plots=hyperparameters["plots"],
        device=hyperparameters["device"],
        amp=hyperparameters["amp"],
        project=train_save_dir,
        name=name
    )

def val_model(val_model_config, data_path, hyperparameters, val_save_dir, split="test", name="val"):
    """
    在测试集上评估 YOLO 模型性能。

    参数:
        val_model_config (YOLO): 已加载的 YOLO 模型。
        data_path (str): 数据集配置文件路径。
        hyperparameters (dict): 超参数配置字典。
        save_dir (str): 评估结果保存目录。
        split (str): 数据集划分，默认为 "test"。
    """
    model = YOLO(val_model_config)
    
    metrics = model.val(
        data=data_path,
        split=split,
        batch=hyperparameters["batch_size"],
        imgsz=hyperparameters["imgsz"],
        save=hyperparameters["save"],
        device=hyperparameters["device"],
        #save_json=True,
        project=val_save_dir,
        name=name
    )

    results_txt_path = os.path.join(val_save_dir, name, "test.txt")
    with open(results_txt_path, 'w') as f:
        f.write(str(metrics))

    results_json_path = os.path.join(val_save_dir, name, "test.json")
    with open(results_json_path, 'w') as f:
        json.dump(metrics, f, default=lambda obj: obj.__dict__ if hasattr(obj, '__dict__') else str(obj), indent=4)

def train_and_val(train_model_config, data_path, hyperparameters, model_name, dataset_name):
    """
    执行模型训练和评估。

    参数:
        train_model_config (str): 模型配置文件路径。
        data_path (str): 数据集配置文件路径。
        hyperparameters (dict): 超参数配置字典。
        model_name (str): 模型名称，用于生成保存目录。
        dataset_name (str): 数据集名称，用于生成保存目录。
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    save_dir = os.path.join("runs", f"{timestamp}_train_{model_name}_{dataset_name}_epochs{hyperparameters['epochs']}")
    train_save_dir = val_save_dir = os.path.join(save_dir)
    train_weights_dir = os.path.join(train_save_dir, "train/weights")
    
    train_model(train_model_config, data_path, hyperparameters, train_save_dir)

    weights = ["best.pt", "last.pt"]
    for weight in weights:
        model_path = os.path.join(train_weights_dir, weight)
        val_model(model_path, data_path, hyperparameters, val_save_dir, split="test", name="val_" + weight.split('.')[0])

# 
if __name__ == '__main__':
    for model_config in model_configs:
        for dataset_name, data_path in datasets.items():
            print(f"开始训练和评估: 模型配置={model_config}, 数据集={dataset_name}")
            train_and_val(model_config, data_path, hyperparameters, model_config.split('.')[0], dataset_name)

'''
# 挂起进程
$timestamp = (Get-Date).ToString("yyyyMMdd_HHmmss"); $p = Start-Process -FilePath python.exe -ArgumentList "train.py" -RedirectStandardOutput "log/train_${timestamp}_1.log" -RedirectStandardError "log/train_${timestamp}_2.log"

nohup bash -c 'python train.py & PID=$!; echo "PID: $PID"; wait $PID' &> "log/train_$(date +%Y%m%d_%H%M%S).log" &
'''
