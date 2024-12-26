import warnings
warnings.filterwarnings('ignore')
import os, tqdm
from ultralytics import YOLO # type: ignore

if __name__ == '__main__':
    yaml_path = "yolo11n-C3k2-WTConv.yaml"
    
    error_result = []
    if 'rtdetr' not in yaml_path and 'cls' not in yaml_path and 'world' not in yaml_path:
        try:
            model = YOLO(yaml_path)
            model.info(detailed=True)
            model.profile([640, 640])
            model.fuse()
        except Exception as e:
            error_result.append(f'{yaml_path} {e}')
    
    for i in error_result:
        print(i)