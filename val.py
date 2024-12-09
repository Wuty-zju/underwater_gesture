from ultralytics import YOLO
from datetime import datetime
import json
import os

datasets = {
    "sample": "datasets/CADDY_gestures_sample/yolo/data.yaml",
    #"complete": "datasets/CADDY_gestures_complete/yolo/data.yaml"
}

hyperparameters = {
    "device": [0],             # 设备: 单GPU ("0"), 多GPU ("0,1"), 或 CPU ("cpu")
    "amp": True,               # 是否启用混合精度训练(默认启用)
    "batch_size": 8,           # 每批次图像数量
    "imgsz": 640,              # 输入图像尺寸
    "save": True,              # 是否保存验证结果
    "plots": True              # 是否生成验证曲线
}

def val_model(val_model_config, data_path, hyperparameters, val_save_dir, split="test", name="val"):
    """
    在测试集上评估 YOLO 模型性能。

    参数:
        val_model_config (str): 模型配置文件路径或权重文件路径。
        data_path (str): 数据集配置文件路径。
        hyperparameters (dict): 超参数配置字典。
        val_save_dir (str): 评估结果保存目录。
        split (str): 数据集划分，默认为 "test"。
        name (str): 评估结果的名称前缀。
    """
    model = YOLO(val_model_config)
    
    metrics = model.val(
        data=data_path,
        split=split,
        batch=hyperparameters["batch_size"],
        imgsz=hyperparameters["imgsz"],
        save=hyperparameters["save"],
        device=hyperparameters["device"],
        project=val_save_dir,
        name=name
    )
    
    results_txt_path = os.path.join(val_save_dir, name, "test.txt")
    with open(results_txt_path, 'w') as f:
        f.write(str(metrics))
    
    results_json_path = os.path.join(val_save_dir, name, "test.json")
    with open(results_json_path, 'w') as f:
        json.dump(metrics, f, default=lambda obj: obj.__dict__ if hasattr(obj, '__dict__') else str(obj), indent=4)

def val(data_path, hyperparameters, dataset_name):
    """
    执行模型验证。

    参数:
        val_model_config (str): 模型配置文件路径或权重文件路径。
        data_path (str): 数据集配置文件路径。
        hyperparameters (dict): 超参数配置字典。
        model_name (str): 模型名称，用于生成保存目录。
        dataset_name (str): 数据集名称，用于生成保存目录。
    """

    save_dir = "runs/202412081830_train_yolo11n-C3k2-EMSC_sample_epochs500"
    train_save_dir = val_save_dir = save_dir
    train_weights_dir = os.path.join(train_save_dir, "train/weights")
    
    weights = ["best.pt", "last.pt"]
    for weight in weights:
        model_path = os.path.join(train_weights_dir, weight)
        val_model(model_path, data_path, hyperparameters, val_save_dir, split="test", name="val_" + weight.split('.')[0])


if __name__ == '__main__':
    for dataset_name, data_path in datasets.items():
        val(data_path, hyperparameters, dataset_name)