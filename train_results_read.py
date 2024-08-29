import os
import yaml
import csv
import re
from tqdm import tqdm

def get_all_params_from_yaml(yaml_path):
    """读取YAML文件并返回所有参数为字典形式"""
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)
        return data if data else {}
    return {}

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
    # 初始表头：保留原有输出内容
    header = ['Train Index', 'Model', 'Batch', 'Epoch', 'Data', 'Path to last.pt', 'Path to best.pt', 'Path to args.yaml', 'Path to results.csv']
    rows = []
    
    pattern = re.compile(r'^train(\d*)$')
    
    # 获取并排序所有训练目录
    train_dirs = sorted(os.listdir(base_directory), key=lambda x: int(pattern.search(x).group(1)) if pattern.search(x) and pattern.search(x).group(1) != '' else 1)

    all_params = set()  # 用于记录所有参数名称

    # 第一次遍历：收集所有参数名称
    for train_dir in train_dirs:
        full_train_dir = os.path.join(base_directory, train_dir)
        if os.path.isdir(full_train_dir):
            args_yaml_path = get_file_path(base_directory, train_dir, 'args.yaml')
            params = get_all_params_from_yaml(args_yaml_path)
            all_params.update(params.keys())

    # 更新头部信息，添加所有参数名称
    header.extend(sorted(all_params))

    # 第二次遍历：写入每个训练的详细信息
    for train_dir in tqdm(train_dirs, desc="Processing directories"):
        full_train_dir = os.path.join(base_directory, train_dir)
        if os.path.isdir(full_train_dir):
            args_yaml_path = get_file_path(base_directory, train_dir, 'args.yaml')
            model, batch, data_value = get_all_params_from_yaml(args_yaml_path).get('model', 'NaN'), get_all_params_from_yaml(args_yaml_path).get('batch', 'NaN'), get_all_params_from_yaml(args_yaml_path).get('data', 'NaN')
            results_csv_path = get_file_path(base_directory, train_dir, 'results.csv')
            epoch = get_final_epoch(results_csv_path)
            last_pt_path = get_file_path(base_directory, train_dir, 'weights/last.pt')
            best_pt_path = get_file_path(base_directory, train_dir, 'weights/best.pt')

            # 提取 data 路径中的文件名
            data_file_name = os.path.basename(data_value).replace('\\', '/') if data_value != 'NaN' else 'NaN'

            # 创建该训练的初始行数据
            row = [train_dir.strip('train') if train_dir.strip('train') != '' else '1', model, batch, epoch, data_file_name, last_pt_path, best_pt_path, args_yaml_path, results_csv_path]

            # 添加所有参数，缺少的使用 'NaN'
            params = get_all_params_from_yaml(args_yaml_path)
            for param in sorted(all_params):
                row.append(params.get(param, 'NaN'))

            rows.append(row)

    # 写入CSV文件
    with open(csv_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # 写入头部
        writer.writerows(rows)  # 写入所有行数据

# 使用函数
base_directory = 'runs/detect'
csv_file_path = 'runs/detect/train_details.csv'
write_csv_and_print_results(base_directory, csv_file_path)
print(f"CSV file created")