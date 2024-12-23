from ultralytics import YOLO # type: ignore
from datetime import datetime
import json
import os

model_configs = [
    #"yolov3n.yaml", "yolov3s.yaml", "yolov3m.yaml", "yolov3l.yaml", "yolov3x.yaml",  
    #"yolov5n.yaml", "yolov5s.yaml", "yolov5m.yaml", "yolov5l.yaml", "yolov5x.yaml",  
    #"yolov6n.yaml", "yolov6s.yaml", "yolov6m.yaml", "yolov6l.yaml", "yolov6x.yaml",  
    #"yolov8n.yaml", "yolov8s.yaml", "yolov8m.yaml", "yolov8l.yaml", "yolov8x.yaml",
    #"yolov9t.yaml", "yolov9s.yaml", "yolov9m.yaml", "yolov9c.yaml", "yolov9e.yaml",
    #"yolov10n.yaml", "yolov10s.yaml", "yolov10m.yaml", "yolov10l.yaml", "yolov10x.yaml",
    #"yolo11n.yaml", "yolo11s.yaml", "yolo11m.yaml", "yolo11l.yaml", "yolo11x.yaml",
    
    #"yolo11-C3k2-AdditiveBlock.yaml",          # sample跑完
    #"yolo11-C3k2-SCConv.yaml",         # sample跑完
    #"yolo11-ReCalibrationFPN-P345.yaml",         # sample跑完
    #"yolo11-WaveletPool.yaml",         # sample跑完
    #"yolo11-efficientViT.yaml",            # sample跑完（缺n/s）
    #"yolo11-C3k2-WTConv.yaml",         # sample跑完
    #"yolo11-LADH.yaml",                # sample跑完
    #"yolo11-RSCD.yaml",                # sample跑完
    #"yolo11-C3k2-EIEM.yaml",           # sample跑完
    #"yolo11-ReCalibrationFPN-P2345.yaml",                # sample跑完
    #"yolo11-FeaturePyramidSharedConv.yaml",                # sample跑完
    #"yolo11-C3k2-LFE.yaml",                # sample跑完
    #"yolo11-C2BRA.yaml",                # sample跑完
    #"yolo11-C2CGA.yaml",                # sample跑完
    #"yolo11-C3k2-MutilScaleEdgeInformationSelect.yaml",                 # sample跑完
    
    #"yolo11-C3k2-KAN.yaml",            # annot access local variable 'kan' where it is not associated with a value(m/l/x)
    #"yolo11-C3k2-RAB.yaml",           # sample跑完 nan（关闭amp，x模型val出错）
    #"yolo11-C3k2-HDRAB.yaml", # 尝试amp-nan（nan,关闭amp，x模型val出错）
    #"yolo11-C2DPB.yaml",  #RuntimeError: The size of tensor a (336) must match the size of tensor b (400) at non-singleton dimension 3
    #"yolo11-C3k2-SFA.yaml", # nan
    #"yolo11x-C3k2-DLKA.yaml", #bash: 第 1 行： 228880 段错误               （核心已转储） python train.py
    #"yolo11x-FDPN-TADDH.yaml", "yolo11l-FDPN-TADDH.yaml", "yolo11m-FDPN-TADDH.yaml", "yolo11s-FDPN-TADDH.yaml", "yolo11n-FDPN-TADDH.yaml",NameError: name 'ModulatedDeformConv2d' is not defined
    #"yolo11n-C3k2-DCNV2-Dynamic.yaml", "yolo11s-C3k2-DCNV2-Dynamic.yaml", "yolo11m-C3k2-DCNV2-Dynamic.yaml", "yolo11l-C3k2-DCNV2-Dynamic.yaml", "yolo11x-C3k2-DCNV2-Dynamic.yaml", bash: 第 1 行： 29787 段错误               （核心已转储） python train.py

    #"yolo11-C3k2-EMSC.yaml",                 # sample跑完
    #"yolo11-C3k2-EMSCP.yaml",                 # sample跑完
    #"yolo11-C3k2-CTA.yaml",                  # sample跑完

    #"yolo11-ReCalibrationFPN-P3456.yaml",        # sample跑完

    #"yolo11-GlobalEdgeInformationTransfer1.yaml",                  # sample跑完
    #"yolo11-GlobalEdgeInformationTransfer2.yaml",                   # sample跑完
    #"yolo11-GlobalEdgeInformationTransfer3.yaml",                   # sample跑完
    #"yolo11-C3k2-MSBlock.yaml",                   # sample跑完
    #"yolo11-ContextGuideFPN.yaml",                  # sample跑完
    
    #"yolo11-C3k2-IDWC.yaml",                  # sample跑完
    #"yolo11-C3k2-IDWB.yaml",                   # sample跑完，对应"yolo11-C3k2-IDWD.yaml"
    #"yolo11-inceptionnext.yaml",                   # sample跑完  
    #"yolo11x-FDPN.yaml",                    # sample跑完
    #"yolo11n-ASF-P2.yaml",                    # sample跑完
    
    #"yolo11x-nmsfree.yaml", "yolo11l-nmsfree.yaml", "yolo11m-nmsfree.yaml", "yolo11s-nmsfree.yaml", "yolo11n-nmsfree.yaml",
    #"yolo11x-goldyolo-asf.yaml", "yolo11l-goldyolo-asf.yaml", "yolo11m-goldyolo-asf.yaml", "yolo11s-goldyolo-asf.yaml", "yolo11n-goldyolo-asf.yaml",

    #"yolo11x-FDPN-DASI.yaml", "yolo11l-FDPN-DASI.yaml", "yolo11m-FDPN-DASI.yaml", "yolo11s-FDPN-DASI.yaml", "yolo11n-FDPN-DASI.yaml",
    #"yolo11x-EIEStem.yaml", "yolo11l-EIEStem.yaml", "yolo11m-EIEStem.yaml", "yolo11s-EIEStem.yaml", "yolo11n-EIEStem.yaml", 
    
    #"yolo11n-C3k2-ContextGuided.yaml", "yolo11s-C3k2-ContextGuided.yaml", "yolo11m-C3k2-ContextGuided.yaml", "yolo11l-C3k2-ContextGuided.yaml", "yolo11x-C3k2-ContextGuided.yaml", 
    #"yolo11n-AIFI.yaml", "yolo11s-AIFI.yaml", "yolo11m-AIFI.yaml", "yolo11l-AIFI.yaml", "yolo11x-AIFI.yaml", 
    #"yolo11n-C3k2-RFAConv.yaml", "yolo11s-C3k2-RFAConv.yaml", "yolo11m-C3k2-RFAConv.yaml", "yolo11l-C3k2-RFAConv.yaml", "yolo11x-C3k2-RFAConv.yaml", 
    
    #"yolo11x-C3k2-Faster.yaml", "yolo11l-C3k2-Faster.yaml", "yolo11m-C3k2-Faster.yaml", "yolo11s-C3k2-Faster.yaml", "yolo11n-C3k2-Faster.yaml",
    #"yolo11x-C3k2-Faster-EMA.yaml", "yolo11l-C3k2-Faster-EMA.yaml", "yolo11m-C3k2-Faster-EMA.yaml", "yolo11s-C3k2-Faster-EMA.yaml", "yolo11n-C3k2-Faster-EMA.yaml",
    #"yolo11n-C3k2-ODConv.yaml", "yolo11s-C3k2-ODConv.yaml", "yolo11m-C3k2-ODConv.yaml", "yolo11l-C3k2-ODConv.yaml", "yolo11x-C3k2-ODConv.yaml", 
    #"yolo11n-C3k2-DBB.yaml", "yolo11s-C3k2-DBB.yaml", "yolo11m-C3k2-DBB.yaml", "yolo11l-C3k2-DBB.yaml", "yolo11x-C3k2-DBB.yaml", 
    #"yolo11n-slimneck.yaml", "yolo11s-slimneck.yaml", "yolo11m-slimneck.yaml", "yolo11l-slimneck.yaml", "yolo11x-slimneck.yaml",
    
    "yolo11x-MAN.yaml", "yolo11l-MAN.yaml", "yolo11m-MAN.yaml", "yolo11s-MAN.yaml", "yolo11n-MAN.yaml", 
    "yolo11x-hyper.yaml", "yolo11l-hyper.yaml", "yolo11m-hyper.yaml", "yolo11s-hyper.yaml", "yolo11n-hyper.yaml", 
    
    
    #"hyper-yolox.yaml", "hyper-yolol.yaml", "hyper-yolom.yaml", "hyper-yolos.yaml", "hyper-yolon.yaml", 
    #"yolo11-C3k2-PConv.yaml", "yolo11-C3k2-PConv.yaml", "yolo11-C3k2-PConv.yaml", "yolo11-C3k2-PConv.yaml", "yolo11-C3k2-PConv.yaml", 
    #"yolo11-atthead.yaml", "yolo11-atthead.yaml", "yolo11-atthead.yaml", "yolo11-atthead.yaml", "yolo11-atthead.yaml", 
    #"yolo11-C3k2-EMA.yaml", "yolo11-C3k2-EMA.yaml", "yolo11-C3k2-EMA.yaml", "yolo11-C3k2-EMA.yaml", "yolo11-C3k2-EMA.yaml", 
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
