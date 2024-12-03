import os
import shutil
import random
import urllib.request
import zipfile
from PIL import Image
import pandas as pd
from tqdm import tqdm

# 全局随机数种子，确保结果可复现
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# 项目目录和数据集目录
PROJECT_DIR = os.path.abspath('')
DATASETS_DIR = os.path.join(PROJECT_DIR, 'datasets')
os.makedirs(DATASETS_DIR, exist_ok=True)

# 数据集信息列表，包括 complete 和 sample 数据集
DATASETS = [
    {
        'name': 'complete',
        'url': 'http://www.caddian.eu/assets/caddy-gestures-TMP/CADDY_gestures_complete_v2_release.zip',
        'expected_file_count': 32860,
        'csv_tp': 'CADDY_gestures_all_true_positives_release_v2.csv',
        'csv_tn': 'CADDY_gestures_all_true_negatives_release_v2.csv',
        'zip_file': 'CADDY_gestures_complete_v2_release.zip',
        'train_ratio': 0.7,
        'val_ratio': 0.2,
        'test_ratio': 0.1,
    },
    {
        'name': 'sample',
        'url': 'http://www.caddian.eu/assets/caddy-gestures-TMP/CADDY_gestures_sample_dataset.zip',
        'expected_file_count': 2108,
        'csv_tp': 'CADDY_gestures_sample_dataset_v2.csv',
        'csv_tn': None,  # sample 数据集没有 TN 数据
        'zip_file': 'CADDY_gestures_sample_dataset.zip',
        'train_ratio': 0.7,
        'val_ratio': 0.2,
        'test_ratio': 0.1,
    }
]

def download_and_extract(dataset_info, use_proxy=False):
    """
    下载并解压指定的数据集。

    参数:
        dataset_info (dict): 包含数据集相关信息的字典。
        use_proxy (bool): 是否使用代理下载，默认为 False。
    """
    base_dir = os.path.join(DATASETS_DIR, f"CADDY_gestures_{dataset_info['name']}")
    zip_path = os.path.join(base_dir, dataset_info['zip_file'])
    extract_to = os.path.join(base_dir, 'raw')

    # 创建基础目录
    os.makedirs(base_dir, exist_ok=True)

    # 使用代理（如果需要）
    if use_proxy:
        proxy = urllib.request.ProxyHandler({'http': 'http://192.168.31.229:7222'})
        opener = urllib.request.build_opener(proxy)
        urllib.request.install_opener(opener)
        print("使用 HTTP 代理进行下载。")

    # 下载数据集 ZIP 文件
    if not os.path.exists(zip_path):
        print(f"正在从 {dataset_info['url']} 下载数据集...")
        response = urllib.request.urlopen(dataset_info['url'])
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 每次读取 1KB

        with open(zip_path, 'wb') as file, tqdm(
            total=total_size, unit='B', unit_scale=True, desc="下载进度", ncols=80
        ) as progress_bar:
            for data in iter(lambda: response.read(block_size), b''):
                file.write(data)
                progress_bar.update(len(data))
        print("下载完成。")
    else:
        print(f"数据集已在 {zip_path} 存在，跳过下载。")

    # 删除并重新创建解压目录
    if os.path.exists(extract_to):
        print(f"删除已有的解压目录 '{extract_to}'...")
        shutil.rmtree(extract_to)
    os.makedirs(extract_to, exist_ok=True)

    # 解压数据集，直接将内容解压到 /raw，不包含顶层文件夹
    print(f"正在将数据集解压到 {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # 获取压缩包中的顶层目录名称
        top_level_dir = zip_ref.namelist()[0].split('/')[0]
        members = [member for member in zip_ref.namelist() if not member.endswith('/')]
        for member in tqdm(members, desc="解压进度", unit="文件", ncols=80):
            # 重构路径，去除顶层目录
            member_path = os.path.relpath(member, top_level_dir)
            target_path = os.path.join(extract_to, member_path)
            # 创建目标目录
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            # 解压文件到目标路径
            with zip_ref.open(member) as source, open(target_path, 'wb') as target:
                shutil.copyfileobj(source, target)
    print("解压完成。")

def process_dataset(dataset_info):
    """
    处理数据集，将图片和标签转换为 YOLO 格式并保存。

    参数:
        dataset_info (dict): 包含数据集相关信息的字典。
    """
    base_dir = os.path.join(DATASETS_DIR, f"CADDY_gestures_{dataset_info['name']}")
    raw_dir = os.path.join(base_dir, 'raw')
    yolo_dir = os.path.join(base_dir, 'yolo')
    images_dir = os.path.join(yolo_dir, 'images')
    labels_dir = os.path.join(yolo_dir, 'labels')

    # 删除并重建 YOLO 目录
    for dir_path in [yolo_dir, images_dir, labels_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)

    # 初始化统计信息
    tp_stats = {'images': 0, 'labels': 0, 'discarded_labels': 0, 'not_found_images': []}
    tn_stats = {'images': 0, 'labels': 0, 'discarded_labels': 0, 'not_found_images': []}
    labels_df = pd.DataFrame()

    # 处理真值（TP）数据集
    tp_csv_path = os.path.join(raw_dir, dataset_info['csv_tp'])
    if os.path.exists(tp_csv_path):
        tp_df = pd.read_csv(tp_csv_path)
        if dataset_info['name'] == 'sample':
            tp_df = tp_df[~tp_df['stereo left'].str.contains('compressed', na=False)]
        labels_df = tp_df[['label name', 'label id']].drop_duplicates()
        tp_stats = convert_annotations(tp_df, 'tp', True, raw_dir, images_dir, labels_dir)
    else:
        print("未找到 TP 数据集或文件不存在。")

    # 处理非真值（TN）数据集
    if dataset_info['csv_tn']:
        tn_csv_path = os.path.join(raw_dir, dataset_info['csv_tn'])
        if os.path.exists(tn_csv_path):
            tn_df = pd.read_csv(tn_csv_path)
            tn_stats = convert_annotations(tn_df, 'tn', False, raw_dir, images_dir, labels_dir)
        else:
            print("未找到 TN 数据集或文件不存在。")

    # 划分数据集
    train_file = os.path.join(yolo_dir, 'train.txt')
    val_file = os.path.join(yolo_dir, 'val.txt')
    test_file = os.path.join(yolo_dir, 'test.txt')
    split_counts = split_dataset(
        images_dir,
        train_file,
        val_file,
        test_file,
        dataset_info['train_ratio'],
        dataset_info['val_ratio'],
        dataset_info['test_ratio']
    )

    # 生成 YAML 配置文件
    generate_yaml(labels_df, yolo_dir, dataset_info)

    # 输出统计信息
    total_images = tp_stats['images'] + tn_stats['images']
    total_labels = tp_stats['labels'] + tn_stats['labels']
    total_discarded = tp_stats['discarded_labels'] + tn_stats['discarded_labels']
    total_not_found = len(tp_stats['not_found_images']) + len(tn_stats['not_found_images'])

    print(f"\n{dataset_info['name']} 数据集处理完成：")
    print(f"  总处理图片数: {total_images}")
    print(f"  总处理标签数: {total_labels}")
    print(f"  丢弃的标签数: {total_discarded}")
    print(f"  未找到的图片数: {total_not_found}")
    print(f"  训练集图片数: {split_counts['train']}")
    print(f"  验证集图片数: {split_counts['val']}")
    print(f"  测试集图片数: {split_counts['test']}")

def convert_annotations(df, prefix, is_tp, raw_dir, images_dir, labels_dir):
    """
    将注释转换为 YOLO 格式并保存。

    参数:
        df (DataFrame): 注释数据。
        prefix (str): 文件名前缀。
        is_tp (bool): 是否为真值数据集。
        raw_dir (str): 原始数据目录。
        images_dir (str): 图片输出目录。
        labels_dir (str): 标签输出目录。

    返回:
        dict: 转换统计信息。
    """
    stats = {'images': 0, 'labels': 0, 'discarded_labels': 0, 'not_found_images': []}

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"处理 {prefix.upper()} 数据集"):
        # 获取左右图像路径
        left_img_path = os.path.join(raw_dir, row['stereo left'].strip('/').replace('/', os.sep))
        right_img_path = os.path.join(raw_dir, row['stereo right'].strip('/').replace('/', os.sep))

        # 检查图像文件是否存在
        if not os.path.exists(left_img_path) or not os.path.exists(right_img_path):
            stats['not_found_images'].extend([left_img_path, right_img_path])
            continue

        # 复制图像到目标目录
        left_target = shutil.copy(left_img_path, os.path.join(images_dir, f"{prefix}_{os.path.basename(left_img_path)}"))
        right_target = shutil.copy(right_img_path, os.path.join(images_dir, f"{prefix}_{os.path.basename(right_img_path)}"))

        # 获取图像尺寸
        try:
            with Image.open(left_target) as img:
                left_width, left_height = img.size
            with Image.open(right_target) as img:
                right_width, right_height = img.size
        except Exception:
            stats['not_found_images'].extend([left_target, right_target])
            continue

        # 处理注释
        if is_tp:
            left_labels = parse_annotations(row.get('roi left'), row['label id'], left_width, left_height)
            right_labels = parse_annotations(row.get('roi right'), row['label id'], right_width, right_height)
            stats['discarded_labels'] += count_invalid_annotations(row.get('roi left')) + count_invalid_annotations(row.get('roi right')) - len(left_labels) - len(right_labels)
        else:
            left_labels = []
            right_labels = []

        # 保存标签文件
        save_labels(left_target, labels_dir, left_labels)
        save_labels(right_target, labels_dir, right_labels)

        # 更新统计信息
        stats['images'] += 2
        stats['labels'] += len(left_labels) + len(right_labels)

    return stats

def parse_annotations(roi_str, label_id, img_width, img_height):
    """
    解析注释字符串并转换为 YOLO 格式。

    参数:
        roi_str (str): ROI 字符串。
        label_id (int): 标签 ID。
        img_width (int): 图像宽度。
        img_height (int): 图像高度。

    返回:
        list: YOLO 格式的注释列表。
    """
    yolo_labels = []
    if isinstance(roi_str, str) and roi_str:
        for roi in roi_str.strip('[]').split(';'):
            try:
                x, y, w, h = map(int, roi.split(','))
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                width_ratio = w / img_width
                height_ratio = h / img_height

                if all(0 <= val <= 1 for val in [x_center, y_center, width_ratio, height_ratio]):
                    yolo_labels.append(f"{label_id} {x_center} {y_center} {width_ratio} {height_ratio}")
            except ValueError:
                continue  # 跳过无法解析的 ROI
    return yolo_labels

def count_invalid_annotations(roi_str):
    """
    统计无效的注释数量。

    参数:
        roi_str (str): ROI 字符串。

    返回:
        int: 无效注释的数量。
    """
    count = 0
    if isinstance(roi_str, str) and roi_str:
        rois = roi_str.strip('[]').split(';')
        count = len(rois)
    return count

def save_labels(image_path, labels_dir, labels):
    """
    保存标签文件。

    参数:
        image_path (str): 图像文件路径。
        labels_dir (str): 标签目录。
        labels (list): 标签列表。
    """
    label_filename = os.path.splitext(os.path.basename(image_path))[0] + '.txt'
    label_path = os.path.join(labels_dir, label_filename)
    with open(label_path, 'w') as f:
        f.write('\n'.join(labels))

def split_dataset(images_dir, train_file, val_file, test_file, train_ratio, val_ratio, test_ratio):
    """
    划分数据集并生成对应的文件列表。

    参数:
        images_dir (str): 图像目录。
        train_file (str): 训练集文件路径。
        val_file (str): 验证集文件路径。
        test_file (str): 测试集文件路径。
        train_ratio (float): 训练集比例。
        val_ratio (float): 验证集比例。
        test_ratio (float): 测试集比例。

    返回:
        dict: 各数据集的图片数量。
    """
    all_images = sorted([os.path.join(images_dir, img) for img in os.listdir(images_dir) if img.endswith('.jpg')])
    paired_images = list(zip(all_images[::2], all_images[1::2]))
    random.shuffle(paired_images)

    total = len(paired_images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        'train': paired_images[:train_end],
        'val': paired_images[train_end:val_end],
        'test': paired_images[val_end:]
    }

    for split_name, split_images in splits.items():
        file_path = {'train': train_file, 'val': val_file, 'test': test_file}[split_name]
        with open(file_path, 'w') as f:
            for left_img, right_img in split_images:
                f.write(f"{os.path.abspath(left_img)}\n")
                f.write(f"{os.path.abspath(right_img)}\n")

    return {k: len(v) * 2 for k, v in splits.items()}  # 每对包含两张图片

def generate_yaml(labels_df, yolo_dir, dataset_info):
    """
    生成 YOLO 数据集的 YAML 配置文件，使用绝对路径。

    参数:
        labels_df (DataFrame): 标签数据。
        yolo_dir (str): YOLO 数据集目录。
        dataset_info (dict): 数据集信息。
    """
    # 确保 names 部分按标签序号从小到大排序
    if not labels_df.empty:
        names_dict = labels_df.set_index('label id')['label name'].to_dict()
        sorted_names = sorted(names_dict.items())  # 按照标签 ID 进行排序
        names_formatted = '\n'.join([f"  {int(k)}: {v}" for k, v in sorted_names])
    else:
        names_formatted = ''

    # 使用绝对路径生成 YAML 内容
    yaml_content = f"""\
# CADDY_gestures_{dataset_info['name']} 数据集 {dataset_info['url']}

path: {os.path.abspath(yolo_dir).replace(os.sep, '/')}
train: {os.path.abspath(os.path.join(yolo_dir, 'train.txt')).replace(os.sep, '/')}
val: {os.path.abspath(os.path.join(yolo_dir, 'val.txt')).replace(os.sep, '/')}
test: {os.path.abspath(os.path.join(yolo_dir, 'test.txt')).replace(os.sep, '/')}

nc: {labels_df['label id'].nunique() if not labels_df.empty else 0}
names:
{names_formatted}
"""

    yaml_path = os.path.join(yolo_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"已生成配置文件 {yaml_path}。")

def main():
    """
    主函数，处理所有数据集。
    """
    for dataset_info in DATASETS:
        print(f"\n开始处理 '{dataset_info['name']}' 数据集...")
        download_and_extract(dataset_info, use_proxy=True)  # 根据需要设置 use_proxy=True
        process_dataset(dataset_info)

    print("\n所有数据集已处理完毕。")

if __name__ == "__main__":
    main()