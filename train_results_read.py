import os
import yaml
import csv
import re
from tqdm import tqdm

def get_model_batch_data_from_yaml(yaml_path):
    """读取YAML文件并返回model值、batch值和data值"""
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)
        model = data.get('model', '未找到model标签')
        batch = data.get('batch', '未找到batch标签')
        data_value = data.get('data', '未找到data标签')
        return model, batch, data_value
    return 'NaN', 'NaN', 'NaN'

def get_file_path(base_directory, train_dir, file_name):
    """检查文件是否存在并返回统一使用正斜杠的相对路径或NaN"""
    full_path = os.path.join(base_directory, train_dir, file_name)
    if os.path.exists(full_path):
        relative_path = os.path.relpath(full_path, os.getcwd())
        return os.path.normpath(relative_path).replace('\\', '/')
    return 'NaN'

def get_final_epoch(results_csv_path):
    """从results.csv文件读取最大epoch值"""
    if os.path.exists(results_csv_path):
        with open(results_csv_path, 'r') as file:
            reader = csv.reader(file)
            next(reader, None)  # 跳过头行
            epochs = [int(row[0].strip()) for row in reader if row and row[0].strip().isdigit()]
            if epochs:
                return max(epochs)
    return 'NaN'

def write_csv_and_print_results(base_directory, csv_path):
    """遍历目录并将信息写入CSV文件"""
    header = ['Train Index', 'Model', 'Batch', 'Epoch', 'Data', 'Path to last.pt', 'Path to best.pt', 'Path to args.yaml', 'Path to results.csv', 'Path to *.0']
    rows = [header]

    pattern = re.compile(r'^train(\d*)$')

    # 获取并排序所有训练目录
    train_dirs = sorted(os.listdir(base_directory), key=lambda x: int(pattern.search(x).group(1)) if pattern.search(x) and pattern.search(x).group(1) != '' else 1)
    
    for train_dir in tqdm(train_dirs, desc="Processing directories"):
        full_train_dir = os.path.join(base_directory, train_dir)
        if os.path.isdir(full_train_dir):
            args_yaml_path = get_file_path(base_directory, train_dir, 'args.yaml')
            model, batch, data_value = get_model_batch_data_from_yaml(args_yaml_path)
            results_csv_path = get_file_path(base_directory, train_dir, 'results.csv')
            epoch = get_final_epoch(results_csv_path)
            last_pt_path = get_file_path(base_directory, train_dir, 'weights/last.pt')
            best_pt_path = get_file_path(base_directory, train_dir, 'weights/best.pt')
            event_file = next((get_file_path(base_directory, train_dir, s) for s in os.listdir(full_train_dir) if s.endswith('.0')), 'NaN')
            
            # 提取 data 路径中的文件名
            data_file_name = os.path.basename(data_value).replace('\\', '/') if data_value != 'NaN' else 'NaN'

            rows.append([train_dir.strip('train') if train_dir.strip('train') != '' else '1', model, batch, epoch, data_file_name, last_pt_path, best_pt_path, args_yaml_path, results_csv_path, event_file])

    # 写入CSV文件
    with open(csv_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

# 使用函数
base_directory = 'runs/detect'
csv_file_path = 'runs/detect/train_details.csv'
write_csv_and_print_results(base_directory, csv_file_path)
print(f"CSV file created")