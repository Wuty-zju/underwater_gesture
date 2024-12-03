import os
import shutil
import pandas as pd
import random
from PIL import Image
from tqdm import tqdm

# 设置训练集-验证集比例
split_ratio = 0.8

# 设置随机数种子
random_seed = 42

# 定义输入文件和输出文件夹路径
project_dir = os.path.normpath('')
datasets_dir = os.path.normpath('datasets')
base_dir = os.path.normpath('datasets/CADDY_gestures_complete')
tn_input_csv = os.path.normpath('datasets/CADDY_gestures_complete/CADDY_gestures_all_true_negatives_release_v2.csv')
tp_input_csv = os.path.normpath('datasets/CADDY_gestures_complete/CADDY_gestures_all_true_positives_release_v2.csv')
output_dir = os.path.normpath('datasets/CADDY_gestures_complete_YOLO/datasets')
output_txt_dir = os.path.normpath('datasets/CADDY_gestures_complete_YOLO')

# 定义训练集和验证集文件路径
train_file = os.path.normpath(os.path.join(output_txt_dir, 'train_complete.txt'))
val_file = os.path.normpath(os.path.join(output_txt_dir, 'val_complete.txt'))

# 定义图像和标签输出目录
images_dir = os.path.normpath(os.path.join(output_dir, 'images'))
labels_dir = os.path.normpath(os.path.join(output_dir, 'labels'))

# 删除并重建输出文件夹
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# 检查并删除已存在的训练集和验证集文件
if os.path.exists(train_file):
    os.remove(train_file)
if os.path.exists(val_file):
    os.remove(val_file)

# 读取真阳性和假阳性数据集
tn_df = pd.read_csv(tn_input_csv)
tp_df = pd.read_csv(tp_input_csv)

# 记录成功转换的图片和标注数量
converted_images = 0
converted_labels = 0
discarded_labels_count = 0

# 记录未找到的图像路径
not_found_images = []

# 处理单个数据集（TN 或 TP）
def process_dataset(df, prefix, is_tp=True):
    global converted_images, converted_labels, discarded_labels_count

    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {prefix.upper()} Dataset"):
        # 获取图像路径
        image_path_left = os.path.normpath(f"{base_dir}/{row['stereo left'].strip().lstrip('/')}")
        image_path_right = os.path.normpath(f"{base_dir}/{row['stereo right'].strip().lstrip('/')}")

        # 检查文件是否存在
        if not os.path.exists(image_path_left):
            not_found_images.append(image_path_left)
            continue
        if not os.path.exists(image_path_right):
            not_found_images.append(image_path_right)
            continue

        # 复制图像到 images 文件夹，并修改名称
        target_path_left = os.path.normpath(shutil.copy(image_path_left, os.path.join(images_dir, f"{prefix}_{os.path.basename(image_path_left)}")))
        target_path_right = os.path.normpath(shutil.copy(image_path_right, os.path.join(images_dir, f"{prefix}_{os.path.basename(image_path_right)}")))

        # 加载图像并获取尺寸
        try:
            with Image.open(target_path_left) as img:
                image_width_left, image_height_left = img.size
            with Image.open(target_path_right) as img:
                image_width_right, image_height_right = img.size
        except FileNotFoundError:
            not_found_images.append(target_path_left)
            not_found_images.append(target_path_right)
            continue

        # 提取ROI信息并处理可能包含多个坐标的情况
        def process_roi(roi_str, width, height):
            if not isinstance(roi_str, str) or not roi_str:
                return []
            roi_parts = roi_str.strip('[]').split(';')
            yolo_lines = []
            for roi_part in roi_parts:
                x, y, w, h = map(int, roi_part.split(','))
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                width_ratio = w / width
                height_ratio = h / height
                # 检查标签的合理性
                if x_center > 1 or y_center > 1 or width_ratio > 1 or height_ratio > 1:
                    global discarded_labels_count
                    discarded_labels_count += 1
                    continue
                yolo_lines.append(f"{row['label id']} {x_center} {y_center} {width_ratio} {height_ratio}")
            return yolo_lines

        # 判断是TP还是TN，选择正确的ROI列
        if is_tp:
            label_lines_left = process_roi(row['roi left'], image_width_left, image_height_left)
            label_lines_right = process_roi(row['roi right'], image_width_right, image_height_right)
        else:
            label_lines_left = []
            label_lines_right = []

        # 使用相同的名称生成标签文件
        label_file_left = os.path.normpath(os.path.join(labels_dir, os.path.splitext(os.path.basename(target_path_left))[0] + '.txt'))
        label_file_right = os.path.normpath(os.path.join(labels_dir, os.path.splitext(os.path.basename(target_path_right))[0] + '.txt'))

        # 保存标签文件
        with open(label_file_left, 'w') as f:
            f.write('\n'.join(label_lines_left) + '\n')

        with open(label_file_right, 'w') as f:
            f.write('\n'.join(label_lines_right) + '\n')

        # 更新成功转换的图片和标注数量
        converted_images += 2
        converted_labels += len(label_lines_left) + len(label_lines_right)

# 处理 TN 和 TP 数据集
process_dataset(tn_df, 'tn', is_tp=False)
process_dataset(tp_df, 'tp', is_tp=True)

# 输出未找到的图片路径和数量
print(f"未匹配的图片数量：{len(not_found_images)}")

# 计算输出文件夹的图片数量和标注文件夹的标签文件数量
image_count = len([f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))])
label_count = len([f for f in os.listdir(labels_dir) if os.path.isfile(os.path.join(labels_dir, f))])

# 输出成功转换的图片和标注数量
print(f"成功转换的图片数量：{converted_images}")
print(f"成功转换的标注数量：{converted_labels}")
print(f"丢弃的标签数量：{discarded_labels_count}")
print(f"YOLO数据集图片数量：{image_count}")
print(f"YOLO数据集标签数量：{label_count}")

# 设置随机数种子
random.seed(random_seed)

# 函数：划分数据集并生成训练和验证集文件
def split_dataset():
    global paired_images, train_images, val_images
    
    # 遍历所有数据集文件并按文件名排序
    all_images = sorted([os.path.normpath(os.path.join(images_dir, f)) for f in os.listdir(images_dir) if f.endswith('.jpg')])

    # 确保双目视觉图像成对出现
    paired_images = list(zip(all_images[::2], all_images[1::2]))

    # 规范化路径
    paired_images = [(os.path.relpath(l, project_dir).replace("\\", "/"), os.path.relpath(r, project_dir).replace("\\", "/")) for l, r in paired_images]

    # 随机划分数据集
    random.shuffle(paired_images)

    # 设置训练集-验证集比例
    split_index = int(split_ratio * len(paired_images))

    # 划分数据集
    train_images = paired_images[:split_index]
    val_images = paired_images[split_index:]

    # 保存训练集和验证集文件
    with open(train_file, 'w') as f:
        for left, right in tqdm(train_images, desc="Saving train images"):
            f.write(f'{left}\n')
            f.write(f'{right}\n')

    with open(val_file, 'w') as f:
        for left, right in tqdm(val_images, desc="Saving val images"):
            f.write(f'{left}\n')
            f.write(f'{right}\n')

# 执行数据集划分函数
split_dataset()

# 计算训练集、验证集以及数据集根目录的相对路径
train_rel_path = os.path.normpath(os.path.relpath(train_file, output_txt_dir)).replace("\\", "/")
val_rel_path = os.path.normpath(os.path.relpath(val_file, output_txt_dir)).replace("\\", "/")
data_rel_path = os.path.normpath(os.path.relpath(output_txt_dir, datasets_dir)).replace("\\", "/")

# 生成 CADDY_gestures.yaml 的内容
labels = tp_df[['label name', 'label id']].drop_duplicates().sort_values('label id')
nc = labels['label id'].max() + 1
names = '\n'.join([f"  {row['label id']}: {row['label name']}" for _, row in labels.iterrows()])

yaml_content = f"""\
# http://www.caddian.eu//CADDY-Underwater-Gestures-Dataset.html
path: {data_rel_path}
train: {train_rel_path}
val: {val_rel_path}

nc: {nc}
names:
{names}
"""

# 将 CADDY_gestures.yaml 写入指定目录
yaml_file = os.path.normpath(os.path.join(output_txt_dir, 'CADDY_gestures_complete.yaml').replace("\\", "/"))
with open(yaml_file, 'w') as f:
    f.write(yaml_content)

print(f"训练集图片数量：{len(train_images) * 2}")
print(f"验证集图片数量：{len(val_images) * 2}")
print(f"训练集标注文件数量：{len(train_images) * 2}")
print(f"验证集标注文件数量：{len(val_images) * 2}")

print("转换完成")