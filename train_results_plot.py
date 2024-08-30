import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

def read_and_filter_details(data_dir, selected_train_indices):
    """
    读取并筛选训练细节数据。

    参数:
    - data_dir: 数据目录路径
    - selected_train_indices: 选择的训练索引列表

    返回:
    - 筛选后的训练细节数据框
    """
    details_csv_path = os.path.join(data_dir, 'runs/detect/train_details.csv')
    details_df = pd.read_csv(details_csv_path)
    return details_df[details_df['Train Index'].isin(selected_train_indices)]

def assign_styles(models):
    """
    为每个模型分配颜色和标记样式。

    参数:
    - models: 模型名称列表

    返回:
    - 模型样式的字典，每个模型对应颜色和标记
    """
    colors = plt.cm.tab20(np.linspace(0, 1, 20))  # 使用tab20颜色映射
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', '+', 'x', 'd', '|', '_', 'H', 'P']
    model_styles = {}
    for idx, model in enumerate(models):
        model_name = model.split('.')[0]  # 确保模型名称与数据一致
        model_styles[model_name] = (colors[idx % len(colors)], markers[idx % len(markers)])
    return model_styles

def plot_individual_plots(details_df, data_dir, plot_dir, model_styles, params, marker_size, line_width, legend_fontsize, dpi):
    """
    绘制单独的训练曲线图。

    参数:
    - details_df: 训练细节数据框
    - data_dir: 数据目录
    - plot_dir: 图像保存目录
    - model_styles: 模型样式字典
    - params: 要绘制的参数列表
    - marker_size: 标记大小
    - line_width: 线条宽度
    - legend_fontsize: 图例字体大小
    - dpi: 图像分辨率
    """
    for _, row in tqdm(details_df.iterrows(), total=details_df.shape[0], desc='Creating Separated plots'):
        train_index = row['Train Index']
        model_name = row['Model'].split('.')[0]
        results_csv_path = row['Path to results.csv']
        
        if pd.notna(results_csv_path) and os.path.exists(os.path.join(data_dir, results_csv_path)):
            results_df = pd.read_csv(os.path.join(data_dir, results_csv_path)).rename(columns=str.strip)
            if 'epoch' not in results_df.columns:
                continue
            
            color, marker = model_styles[model_name]
            individual_plot_dir = os.path.join(plot_dir, f'train{train_index}')
            os.makedirs(individual_plot_dir, exist_ok=True)
            
            for param in params:
                if param in results_df.columns:
                    plt.figure(figsize=(10, 6))
                    plt.plot(results_df['epoch'], results_df[param], label=model_name, marker=marker,
                             markersize=marker_size, linewidth=line_width, color=color)
                    plt.xlabel('Epoch')
                    plt.ylabel(param)
                    plt.title(f'{model_name} - {param}')
                    plt.legend(fontsize=legend_fontsize)
                    plt.savefig(os.path.join(individual_plot_dir, f'{param.replace("/", "_")}.png'), dpi=dpi)
                    plt.close()

def collect_plot_data(details_df, data_dir, model_styles, params, log_scale_threshold):
    """
    收集用于绘制比较图的数据，并为每个大类分配线型。

    参数:
    - details_df: 训练细节数据框
    - data_dir: 数据目录
    - model_styles: 模型样式字典
    - params: 要绘制的参数列表
    - log_scale_threshold: 使用对数刻度的阈值

    返回:
    - plots_data: 绘图数据字典
    - use_log_scale: 是否使用对数刻度的字典
    """
    plots_data = {param: [] for param in params}
    use_log_scale = {param: False for param in params}

    # 定义大类与线型的映射
    category_line_styles = {
        '小型数据集-旧-预训练-100': '-',
        '小型数据集-旧-预训练-500': '--',
        '小型数据集-旧-预训练-2000': ':',
        '大型数据集-新-预训练-100': '-.',
        '小型数据集-新-预训练-100': '-',
        '小型数据集-新-无预训练-100': '--'
    }
    
    # 根据训练索引分配大类
    def get_category(train_index):
        if train_index in [12, 13, 14, 15, 17, 19, 27, 28, 29, 30, 31, 32]:
            return '小型数据集-旧-预训练-100'
        elif train_index in [36, 37, 40, 41, 43, 51, 52, 54, 55, 56, 57, 58, 62, 63, 65, 66]:
            return '小型数据集-旧-预训练-500'
        elif train_index in [69, 70, 71, 72, 73]:
            return '小型数据集-旧-预训练-2000'
        elif train_index in [75, 76, 81, 82, 84, 87, 88, 91, 93, 94, 95, 97, 101, 102, 103, 104, 105, 106, 107]:
            return '大型数据集-新-预训练-100'
        elif train_index in [98, 99, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121]:
            return '小型数据集-新-预训练-100'
        elif train_index in [122, 123, 124, 125, 127, 128, 130, 131, 132, 133, 134, 136, 138, 139, 140, 142, 143, 144]:
            return '小型数据集-新-无预训练-100'
        else:
            return '未知'

    for _, row in tqdm(details_df.iterrows(), total=details_df.shape[0], desc='Collecting data for compared plots'):
        model_name = row['Model'].split('.')[0]
        results_csv_path = row['Path to results.csv']
        train_index = row['Train Index']
        
        if pd.notna(results_csv_path) and os.path.exists(os.path.join(data_dir, results_csv_path)):
            results_df = pd.read_csv(os.path.join(data_dir, results_csv_path)).rename(columns=str.strip)
            if 'epoch' not in results_df.columns:
                continue
            
            color, marker = model_styles[model_name]
            category = get_category(train_index)
            line_style = category_line_styles.get(category, '-')
            
            for param in params:
                if param in results_df.columns:
                    values = results_df[param]
                    epochs = results_df['epoch']
                    plots_data[param].append((epochs, values, model_name, color, marker, line_style))
                    
                    # 对数刻度判断（如果需要）
                    # if values.max() / values.min() > log_scale_threshold:
                    #    use_log_scale[param] = True

    return plots_data, use_log_scale

def plot_combined_plots(plots_data, use_log_scale, plot_dir, dpi, legend_fontsize):
    """
    绘制合并的比较图。

    参数:
    - plots_data: 绘图数据字典
    - use_log_scale: 是否使用对数刻度的字典
    - plot_dir: 图像保存目录
    - dpi: 图像分辨率
    - legend_fontsize: 图例字体大小
    """
    compared_dir = os.path.join(plot_dir, 'compared')
    os.makedirs(compared_dir, exist_ok=True)
    
    for param, data in tqdm(plots_data.items(), desc='Creating Compared plots'):
        plt.figure(figsize=(10, 6))
        legend_dict = OrderedDict()
        for epochs, values, model_name, color, marker, line_style in data:
            line, = plt.plot(epochs, values, label=model_name, linewidth=1.0, color=color, linestyle=line_style)
            plt.text(epochs.iloc[-1], values.iloc[-1], model_name, fontsize=1, color=color)
            legend_dict[model_name] = line
        plt.xlabel('Epoch')
        plt.ylabel(param)
        plt.title(param)
        if use_log_scale[param]:
            plt.yscale('log')
        plt.legend(legend_dict.values(), legend_dict.keys(), fontsize=legend_fontsize, loc='upper left')
        plt.savefig(os.path.join(compared_dir, f'{param.replace("/", "_")}_compared.png'), dpi=dpi)
        plt.close()

def create_plots(data_dir, plot_dir, marker_size=0.5, line_width=0.5, legend_fontsize=5, dpi=1000, log_scale_threshold=100, plot_mode=1, selected_train_indices=None):
    """
    创建并绘制图像。

    参数:
    - data_dir: 数据目录
    - plot_dir: 图像保存目录
    - marker_size: 标记大小
    - line_width: 线条宽度
    - legend_fontsize: 图例字体大小
    - dpi: 图像分辨率
    - log_scale_threshold: 对数刻度阈值
    - plot_mode: 绘图模式
    - selected_train_indices: 选择的训练索引列表
    """
    details_df = read_and_filter_details(data_dir, selected_train_indices)
    print("Filtered details_df:", details_df)
    
    params = ['train/box_loss', 'train/cls_loss', 'train/dfl_loss',
              'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)',
              'metrics/mAP50-95(B)', 'val/box_loss', 'val/cls_loss', 'val/dfl_loss',
              'lr/pg0', 'lr/pg1', 'lr/pg2']
    
    model_styles = assign_styles(details_df['Model'].unique())
    
    if plot_mode in [0, 2]:
        plot_individual_plots(details_df, data_dir, plot_dir, model_styles, params, marker_size, line_width, legend_fontsize, dpi)
    
    if plot_mode in [0, 1, 3]:
        plots_data, use_log_scale = collect_plot_data(details_df, data_dir, model_styles, params, log_scale_threshold)
        plot_combined_plots(plots_data, use_log_scale, plot_dir, dpi, legend_fontsize)

# 配置目录和训练索引
data_dir = './'
plot_output_dir = './plots'

# 定义选定的训练索引数组
# 小型数据集-旧-预训练-100 [12, 13, 14, 15, 17, 19, 27, 28, 29, 30, 31, 32]
# 小型数据集-旧-预训练-500 [36, 37, 40, 41, 43, 51, 52, 54, 55, 56, 57, 58, 62, 63, 65, 66]
# 小型数据集-旧-预训练-2000 [69, 70, 71, 72, 73]
# 大型数据集-新-预训练-100 [75, 76, 81, 82, 84, 87, 88, 91, 93, 94, 95, 97, 101, 102, 103, 104, 105, 106, 107]
# 小型数据集-新-预训练-100 [98, 99, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121]
# 小型数据集-新-无预训练-100 [122, 123, 124, 125, 127, 128, 130, 131, 132, 133, 134, 136, 138, 139, 140, 142, 143, 144]
selected_train_indices = [12, 13, 14, 15, 17, 19, 27, 28, 29, 30, 31, 32,
                          98, 99, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
                          122, 123, 124, 125, 127, 128, 130, 131, 132, 133, 134, 136, 138, 139, 140, 142, 143, 144]

# 生成图像
create_plots(data_dir, plot_output_dir, marker_size=0.5, line_width=0.5, legend_fontsize=5, dpi=1500, plot_mode=3, selected_train_indices=selected_train_indices)