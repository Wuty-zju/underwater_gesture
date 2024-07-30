import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

def create_plots(data_dir, plot_dir, marker_size=0.5, line_width=0.5, legend_fontsize=5, dpi=1000, log_scale_threshold=100, plot_mode=1, random_seed=42):
    # 构建 train_details.csv 文件路径
    details_csv_path = os.path.join(data_dir, 'runs/detect/train_details.csv')

    # 读取 train_details.csv 文件
    details_df = pd.read_csv(details_csv_path)

    # 定义需要绘制的参数
    params = ['train/box_loss', 'train/cls_loss', 'train/dfl_loss',
              'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)',
              'metrics/mAP50-95(B)', 'val/box_loss', 'val/cls_loss', 'val/dfl_loss',
              'lr/pg0', 'lr/pg1', 'lr/pg2']

    # 初始化存储绘图数据的字典
    plots_data = {param: [] for param in params}
    plots_data_last_50_percent = {param: [] for param in params}
    use_log_scale = {param: False for param in params}

    # 定义可用的标记样式和颜色
    np.random.seed(random_seed)
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', '+', 'x', 'd', '|', '_', 'H', 'P']
    
    # 随机分配颜色和标记
    model_styles = {}
    color_idx = 0

    # 遍历 train_details.csv 中的每一行并显示进度条
    for index, row in tqdm(details_df.iterrows(), total=details_df.shape[0], desc='Creating Separated plots'):
        train_index = row['Train Index']
        model_name = row['Model'].split('.')[0]  # 提取 Model 的名称
        results_csv_path = row['Path to results.csv']
        epoch_column = 'epoch'

        # 检查 results.csv 路径是否有效且存在
        if pd.notna(results_csv_path) and os.path.exists(os.path.join(data_dir, results_csv_path)):
            results_df = pd.read_csv(os.path.join(data_dir, results_csv_path))

            # 去除列名中的前后空格
            results_df.columns = results_df.columns.str.strip()

            # 检查 epoch 列是否存在
            if epoch_column not in results_df.columns:
                continue

            # 为模型分配颜色和标记样式
            if model_name not in model_styles:
                model_styles[model_name] = (colors[color_idx % len(colors)], markers[color_idx % len(markers)])
                color_idx += 1

            color, marker = model_styles[model_name]

            # 如果 plot_mode 为0或2，则绘制分开图
            if plot_mode in [0, 2]:
                individual_plot_dir = os.path.join(plot_dir, f'train{train_index}')
                os.makedirs(individual_plot_dir, exist_ok=True)
                
                # 为每个参数创建单独的折线图
                for param in params:
                    if param in results_df.columns and results_df[epoch_column].max() >= 100:  # 检查步长是否大于等于100
                        plt.figure(figsize=(10, 6))
                        plt.plot(results_df[epoch_column], results_df[param], label=f'{model_name}', marker=marker,
                                 markersize=marker_size, linewidth=line_width, color=color)
                        plt.xlabel('Epoch')
                        plt.ylabel(param)
                        plt.title(f'{model_name} - {param}')
                        plt.legend(fontsize=legend_fontsize)
                        individual_plot_filename = os.path.join(individual_plot_dir, f'{param.replace("/", "_")}.png')
                        plt.savefig(individual_plot_filename, dpi=dpi)
                        plt.close()

    # 如果 plot_mode 为0或1，则创建每个参数的合并图，并收集数据
    if plot_mode in [0, 1]:
        compared_dir = os.path.join(plot_dir, 'compared')
        os.makedirs(compared_dir, exist_ok=True)
        
        for index, row in tqdm(details_df.iterrows(), total=details_df.shape[0], desc='Collecting data for compared plots'):
            model_name = row['Model'].split('.')[0]
            results_csv_path = row['Path to results.csv']
            epoch_column = 'epoch'

            # 检查 results.csv 路径是否有效且存在
            if pd.notna(results_csv_path) and os.path.exists(os.path.join(data_dir, results_csv_path)):
                results_df = pd.read_csv(os.path.join(data_dir, results_csv_path))

                # 去除列名中的前后空格
                results_df.columns = results_df.columns.str.strip()

                # 检查 epoch 列是否存在
                if epoch_column not in results_df.columns:
                    continue

                color, marker = model_styles[model_name]

                # 为每个参数收集数据
                for param in params:
                    if param in results_df.columns and results_df[epoch_column].max() >= 100:  # 检查步长是否大于等于500
                        values = results_df[param]
                        epochs = results_df[epoch_column]

                        plots_data[param].append((epochs, values, model_name, color, marker))

                        # 收集后50%步长的数据
                        mid_point = len(values) // 2
                        plots_data_last_50_percent[param].append((epochs[mid_point:], values[mid_point:], model_name, color, marker))

                        # 检查数据范围以确定是否使用对数纵坐标
                        if values.max() / values.min() > log_scale_threshold:
                            use_log_scale[param] = True

        # 自定义排序
        model_order = ['YOLOv8', 'YOLOv9', 'YOLOv10']
        
        # 创建合并图
        for param, data in tqdm(plots_data.items(), desc='Creating Compared plots'):
            plt.figure(figsize=(10, 6))
            legend_dict = OrderedDict()
            for epochs, values, model_name, color, marker in data:
                trimmed_model_name = model_name[4:] if len(model_name) > 4 else model_name  # 去掉前四个字母
                line, = plt.plot(epochs, values, label=f'{model_name}', linewidth=line_width, color=color)
                plt.text(epochs.iloc[-1], values.iloc[-1], trimmed_model_name, fontsize=1, color=color)
                if model_name in model_order and model_name not in legend_dict:
                    legend_dict[model_name] = line
            plt.xlabel('Epoch')
            plt.ylabel(param)
            plt.title(param)
            if use_log_scale[param]:
                plt.yscale('log')
            plt.legend(legend_dict.values(), legend_dict.keys(), fontsize=legend_fontsize, loc='upper left')
            
            plot_filename = os.path.join(compared_dir, f'{param.replace("/", "_")}_compared.png')
            plt.savefig(plot_filename, dpi=dpi)
            plt.close()

        # 创建后50%步长的合并图
        compared_last_50_percent_dir = os.path.join(plot_dir, 'compared_last_50_percent')
        os.makedirs(compared_last_50_percent_dir, exist_ok=True)
        
        for param, data in tqdm(plots_data_last_50_percent.items(), desc='Creating Compared Last 50 Percent plots'):
            plt.figure(figsize=(10, 6))
            legend_dict = OrderedDict()
            for epochs, values, model_name, color, marker in data:
                trimmed_model_name = model_name[4:] if len(model_name) > 4 else model_name  # 去掉前四个字母
                line, = plt.plot(epochs, values, label=f'{model_name}', linewidth=line_width, color=color)
                plt.text(epochs.iloc[-1], values.iloc[-1], trimmed_model_name, fontsize=1, color=color)
                if model_name in model_order and model_name not in legend_dict:
                    legend_dict[model_name] = line
            plt.xlabel('Epoch')
            plt.ylabel(param)
            plt.title(f'Last 50% - {param}')
            if use_log_scale[param]:
                plt.yscale('log')
            plt.legend(legend_dict.values(), legend_dict.keys(), fontsize=legend_fontsize, loc='upper left')
            
            plot_filename = os.path.join(compared_last_50_percent_dir, f'{param.replace("/", "_")}_compared_last_50_percent.png')
            plt.savefig(plot_filename, dpi=dpi)
            plt.close()

# 定义数据目录和图像输出目录
data_dir = './'
plot_output_dir = './plots'

# 调用函数生成图表
create_plots(data_dir, plot_output_dir, marker_size=0.5, line_width=0.5, legend_fontsize=5, dpi=2000, plot_mode=0, random_seed=42)