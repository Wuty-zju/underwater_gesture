from ultralytics import YOLO # type: ignore
from datetime import datetime
import json
import os

model_configs = [
    
    ## Official
    #"yolov3n.yaml", "yolov3s.yaml", "yolov3m.yaml", "yolov3l.yaml", "yolov3x.yaml",  
    #"yolov5n.yaml", "yolov5s.yaml", "yolov5m.yaml", "yolov5l.yaml", "yolov5x.yaml",  
    #"yolov6n.yaml", "yolov6s.yaml", "yolov6m.yaml", "yolov6l.yaml", "yolov6x.yaml",  
    #"yolov8n.yaml", "yolov8s.yaml", "yolov8m.yaml", "yolov8l.yaml", "yolov8x.yaml",
    #"yolov9t.yaml", "yolov9s.yaml", "yolov9m.yaml", "yolov9c.yaml", "yolov9e.yaml",
    #"yolov10n.yaml", "yolov10s.yaml", "yolov10m.yaml", "yolov10l.yaml", "yolov10x.yaml",
    #"yolo11n.yaml", "yolo11s.yaml", "yolo11m.yaml", "yolo11l.yaml", "yolo11x.yaml",
    
    ## BackBone
    #"yolo11-efficientViT.yaml",                # 1 # BackBone # sample跑完
    #"yolo11-fasternet.yaml",                   # 2 # BackBone # sample跑完 # Warning-需要修改路径 cd ultralytics-8.3.9 && "sample": "../datasets/CADDY_gestures_sample/yolo/data.yaml", && nohup bash -c 'python ../train.py & PID=$!; echo "PID: $PID"; wait $PID' &> "../log/train_$(date +%Y%m%d_%H%M%S).log" &
    #"yolo11-timm.yaml",                        # 3 # BackBone # sample跑完
    #"yolo11-convnextv2.yaml",                  # 4 # BackBone # sample跑完
    #"yolo11-EfficientFormerV2.yaml",           # 9 # BackBone  # Error-5
    #"yolo11-vanillanet.yaml",                  # 16 # BackBone # sample跑完
    #"yolo11l-RevCol.yaml", "yolo11m-RevCol.yaml", "yolo11s-RevCol.yaml", "yolo11n-RevCol.yaml",                     # 18 # BackBone
    #"yolo11x-LSKNet.yaml", "yolo11l-LSKNet.yaml", "yolo11m-LSKNet.yaml", "yolo11s-LSKNet.yaml", "yolo11n-LSKNet.yaml",                       # 19 # BackBone
    #"yolo11-swintransformer.yaml",             # 39 # BackBone
    #"yolo11-repvit.yaml",                      # 40 # BackBone
    
    
    

    ## SPPF
    #"yolo11n-FocalModulation.yaml", "yolo11s-FocalModulation.yaml", "yolo11m-FocalModulation.yaml", "yolo11l-FocalModulation.yaml", "yolo11x-FocalModulation.yaml",              # 36 # SPPF
    
    ## Neck
    #"yolo11-bifpn.yaml",                       # 6 # Neck 
    #"yolo11-slimneck.yaml",                    # 13 # Neck # sample跑完
    #"yolo11-AFPN-P345.yaml",                   # 15-a # Neck
    #"yolo11-AFPN-P345-Custom",                 # 15-b # Neck
    #"yolo11-AFPN-P2345.yaml",                  # 15-c # Neck
    #"yolo11-AFPN-P2345-Custom",                # 15-d # Neck
    #"yolo11-RCSOSA.yaml",                      # 26 # Neck
    #"yolo11-goldyolo.yaml",                    # 43 # Neck
    #"yolo11-GFPN.yaml",                        # 48 # Neck
    #"yolo11-EfficientRepBiPAN.yaml",           # 50 # Neck
    

    
    ## Head
    #"yolo11-dyhead.yaml",                      # 5 # Head 
    #"yolo11-EfficientHead.yaml",               # 9 # Head
    #"yolo11-aux.yaml",                         # 32 # Head
    #"yolo11-dyhead-DCNV3.yaml",                # 35 # Head
    
    ## PostProcess
    
    ## UpSample & DownSample
    #"yolo11-LAWDS.yaml",                       # 23 # DownSample
    #"yolo11-ContextGuidedDown.yaml",           # 45 # DownSample
    #"yolo11-SPDConv.yaml",                     # 49 # DownSample
    
    ## C3k2
    #"yolo11-C3k2-Faster.yaml",                 # 7 # C3k2 # sample跑完
    #"yolo11-C3k2-ODConv.yaml",                 # 8 # C3k2 # sample跑完
    #"yolo11-C3k2-Faster-EMA.yaml",             # 10 # C3k2 # sample跑完
    #"yolo11-C3k2-DBB.yaml",                    # 11 # C3k2 # sample跑完
    #"yolo11-C3k2-CloAtt.yaml",                 # 17 # C3k2 # Error-5
    #"yolo11-C3k2-SCConv.yaml",                 # 20 # C3k2 # sample跑完
    #"yolo11-C3k2-SCcConv.yaml",                # 21 # C3k2
    #"yolo11-C3k2-EMSC.yaml",                   # 24 # C3k2 # sample跑完
    #"yolo11-C3k2-EMSCP.yaml",                  # 25 # C3k2 # sample跑完
    #"yolo11-KernelWarehouse.yaml",             # 27 # C3k2 
    #"yolo11-C3k2-DySnakeConv.yaml",            # 30 # C3k2
    #"yolo11-C3k2-DCNV2.yaml",                  # 33 # C3k2 
    #"yolo11-C3k2-DCNV3.yaml",                  # 34 # C3K2 
    #"yolo11-C3k2-OREPA.yaml",                  # 37 # C3K2
    #"yolo11-C3k2-REPVGGOREPA.yaml",            # 38 # C3K2
    #"yolo11-C3k2-DCNV2-Dynamic.yaml",          # 42 # C3K2 # bash: 第 1 行： 29787 段错误               （核心已转储） python train.py
    #"yolo11-C3k2-ContextGuided.yaml",          # 44 # C3K2 # sample跑完
    #"yolo11-C3k2-MSBlock.yaml",                # 46 # C3K2 # sample跑完
    #"yolo11-C3k2-DLKA.yaml",                   # 47 # C3K2 # bash: 第 1 行： 228880 段错误               （核心已转储） python train.py
    
    ## C2PSA
    
    ## Mixup
    #"yolo11-fasternet-bifpn.yaml",             # 41 # Mixup # BackBone # Neck # Warning-可能修改路径
    
    ## Attantion
    #"yolo11-attention.yaml",                   # 14 # Attention # Warning-需要修改yaml
    
    ## Label Assign
    #"Adaptive Training Sample Selection",      # 12 # Label Assign
    
    ## Loss
    #"MPDiou",                                  # 22 # Loss      
    #“Normalized Gaussian Wasserstein Distance”,# 28 # Loss # Label Assign
    #"SlideLoss and EMASlideLoss",              # 29 # Loss 
    
    
    
    #"yolo11-C3k2-AdditiveBlock.yaml",          # sample跑完
    #"yolo11-ReCalibrationFPN-P345.yaml",         # sample跑完
    #"yolo11-WaveletPool.yaml",         # sample跑完
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
    #"yolo11-C3k2-CTA.yaml",                  # sample跑完
    #"yolo11-ReCalibrationFPN-P3456.yaml",        # sample跑完
    #"yolo11-GlobalEdgeInformationTransfer1.yaml",                  # sample跑完
    #"yolo11-GlobalEdgeInformationTransfer2.yaml",                   # sample跑完
    #"yolo11-GlobalEdgeInformationTransfer3.yaml",                   # sample跑完
    #"yolo11-ContextGuideFPN.yaml",                  # sample跑完
    #"yolo11-C3k2-IDWC.yaml",                  # sample跑完
    #"yolo11-C3k2-IDWB.yaml",                   # sample跑完，对应"226.yolo11-C3k2-IDWD.yaml"
    #"yolo11-inceptionnext.yaml",                   # sample跑完  
    #"yolo11-FDPN.yaml",                    # sample跑完
    #"yolo11-ASF-P2.yaml",                    # sample跑完
    #"yolo11-nmsfree.yaml",                    # sample跑完
    #"yolo11-goldyolo-asf.yaml",                    # sample跑完
    #"yolo11-FDPN-DASI.yaml",                    # sample跑完
    #"yolo11-EIEStem.yaml",                     # sample跑完
    #"yolo11-AIFI.yaml",                     # sample跑完
    #"yolo11-C3k2-RFAConv.yaml",                     # sample跑完
    #"yolo11-MAN.yaml",                      # sample跑完
    #"yolo11-hyper.yaml",                      # sample跑完
    #"hyper-yolo.yaml",                      # sample跑完
    #"yolo11-C3k2-PConv.yaml",                      # sample跑完
    #"yolo11-atthead.yaml",                      # sample跑完
    #"yolo11-C3k2-EMA.yaml",                      # sample跑完
    #"yolo11-HSFPN.yaml",                       # sample跑完
    #"yolo11-HSPAN.yaml",                       # sample跑完
    #"yolo11x-MAN-Faster.yaml",                 # sample跑完
    #"yolo11x-MAN-FasterCGLU.yaml",           # sample跑完
    #"yolo11x-MAN-Star.yaml",                  # sample跑完
    #"yolo11n-msga.yaml",                     # sample跑完
    #"yolo11n-MutilBackbone-MSGA.yaml",            # sample跑完
]

'''
    #"yolo11-C3k2-KAN.yaml",            # annot access local variable 'kan' where it is not associated with a value(m/l/x)
    #"yolo11-C3k2-RAB.yaml",           # sample跑完 nan(关闭amp, x模型val出错)
    #"yolo11-C3k2-HDRAB.yaml", # 尝试amp-nan（nan,关闭amp，x模型val出错）
    #"yolo11-C2DPB.yaml",  #RuntimeError: The size of tensor a (336) must match the size of tensor b (400) at non-singleton dimension 3
    #"yolo11-C3k2-SFA.yaml", # nan
    #"yolo11-FDPN-TADDH.yaml",       NameError: name 'ModulatedDeformConv2d' is not defined
    #"yolo11-C3k2-DCNV4.yaml",  #关闭amp(x模型 nan)
'''

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
