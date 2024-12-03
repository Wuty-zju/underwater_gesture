import pandas as pd
import os
import shutil
import tensorflow as tf
from tqdm import tqdm

# 设置TensorFlow日志级别以避免警告和信息日志的输出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 忽略所有TensorFlow信息性消息，只显示错误
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # 仅显示错误信息

def clean_output_directory(output_dir):
    """清空输出目录中的所有文件和文件夹。"""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # 删除整个目录
    os.makedirs(output_dir)  # 重新创建目录

def extract_tags_to_csv(relative_event_path, output_csv_path, interested_tags):
    """从事件文件中提取感兴趣的标签并保存到CSV文件中。"""
    events = []  # 存储所有事件的列表

    for e in tf.compat.v1.train.summary_iterator(relative_event_path):
        for v in e.summary.value:
            if v.tag in interested_tags:
                events.append({
                    "step": e.step,
                    "tag": v.tag,
                    "value": v.simple_value,
                    "wall_time": e.wall_time
                })

    if events:
        df_events = pd.DataFrame(events)
        pivot_df = df_events.pivot_table(index='step', columns='tag', values='value', aggfunc='first')
        pivot_df.to_csv(output_csv_path, index=True)
    else:
        df_empty = pd.DataFrame(columns=list(interested_tags))
        df_empty.to_csv(output_csv_path, index=False)

def auto_collect_tags(relative_event_path):
    """自动收集事件文件中出现的所有标签。"""
    all_tags = set()  # 存储所有标签的集合
    for e in tf.compat.v1.train.summary_iterator(relative_event_path):
        for v in e.summary.value:
            all_tags.add(v.tag)
    return all_tags

def process_event_files():
    """处理事件文件并生成对应的CSV输出。"""
    df = pd.read_csv('./runs/detect/train_details.csv')
    script_dir = os.path.dirname(__file__)  # 获取脚本目录
    output_dir = os.path.join(script_dir, 'others/events_out_tfevents')  # 定义输出目录
    clean_output_directory(output_dir)  # 清空并重建输出目录

    # 使用tqdm包装循环，以显示进度条
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="处理进度"):
        relative_path = os.path.join(script_dir, row['Path to *.0'])  # 构建完整的事件文件路径
        interested_tags = auto_collect_tags(relative_path)  # 收集感兴趣的标签
        output_path = os.path.join(output_dir, f'results_train{row["Train Index"]}.csv')  # 定义输出文件路径
        extract_tags_to_csv(relative_path, output_path, interested_tags)  # 提取标签并保存为CSV文件

    print(f"所有数据已保存在目录: {output_dir}")  # 在所有处理完成后输出目录路径

if __name__ == '__main__':
    process_event_files()
