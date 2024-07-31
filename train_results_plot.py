import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

def read_and_filter_details(data_dir, selected_train_indices):
    details_csv_path = os.path.join(data_dir, 'runs/detect/train_details.csv')
    details_df = pd.read_csv(details_csv_path)
    return details_df[details_df['Train Index'].isin(selected_train_indices)]

def assign_styles(models):
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', '+', 'x', 'd', '|', '_', 'H', 'P']
    model_styles = {}
    color_idx = 0
    for model in models:
        model_name = model.split('.')[0]  # Ensure model name is consistent with data
        model_styles[model_name] = (colors[color_idx % len(colors)], markers[color_idx % len(markers)])
        color_idx += 1
    return model_styles

def plot_individual_plots(details_df, data_dir, plot_dir, model_styles, params, marker_size, line_width, legend_fontsize, dpi):
    for index, row in tqdm(details_df.iterrows(), total=details_df.shape[0], desc='Creating Separated plots'):
        train_index = row['Train Index']
        model_name = row['Model'].split('.')[0]
        results_csv_path = row['Path to results.csv']
        epoch_column = 'epoch'
        
        if pd.notna(results_csv_path) and os.path.exists(os.path.join(data_dir, results_csv_path)):
            results_df = pd.read_csv(os.path.join(data_dir, results_csv_path))
            results_df.columns = results_df.columns.str.strip()
            
            if epoch_column not in results_df.columns:
                continue
            
            color, marker = model_styles[model_name]
            individual_plot_dir = os.path.join(plot_dir, f'train{train_index}')
            os.makedirs(individual_plot_dir, exist_ok=True)
            
            for param in params:
                if param in results_df.columns:
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

def collect_plot_data(details_df, data_dir, model_styles, params, log_scale_threshold):
    plots_data = {param: [] for param in params}
    use_log_scale = {param: False for param in params}
    
    for index, row in tqdm(details_df.iterrows(), total=details_df.shape[0], desc='Collecting data for compared plots'):
        model_name = row['Model'].split('.')[0]
        results_csv_path = row['Path to results.csv']
        epoch_column = 'epoch'
        
        if pd.notna(results_csv_path) and os.path.exists(os.path.join(data_dir, results_csv_path)):
            results_df = pd.read_csv(os.path.join(data_dir, results_csv_path))
            results_df.columns = results_df.columns.str.strip()
            
            if epoch_column not in results_df.columns:
                continue
            
            color, marker = model_styles[model_name]
            
            for param in params:
                if param in results_df.columns:
                    values = results_df[param]
                    epochs = results_df[epoch_column]
                    plots_data[param].append((epochs, values, model_name, color, marker))
                    
                    if values.max() / values.min() > log_scale_threshold:
                        use_log_scale[param] = True
    
    return plots_data, use_log_scale

def plot_combined_plots(plots_data, use_log_scale, plot_dir, line_width, legend_fontsize, dpi):
    compared_dir = os.path.join(plot_dir, 'compared')
    os.makedirs(compared_dir, exist_ok=True)
    
    for param, data in tqdm(plots_data.items(), desc='Creating Compared plots'):
        plt.figure(figsize=(10, 6))
        legend_dict = OrderedDict()
        for epochs, values, model_name, color, marker in data:
            trimmed_model_name = model_name[4:] if len(model_name) > 4 else model_name
            line, = plt.plot(epochs, values, label=f'{model_name}', linewidth=line_width, color=color)
            plt.text(epochs.iloc[-1], values.iloc[-1], trimmed_model_name, fontsize=1, color=color)
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

def create_plots(data_dir, plot_dir, marker_size=0.5, line_width=0.5, legend_fontsize=5, dpi=1000, log_scale_threshold=100, plot_mode=1, selected_train_indices=None):
    # Read and filter data
    details_df = read_and_filter_details(data_dir, selected_train_indices)
    print("Filtered details_df:")
    print(details_df)
    
    # Define parameters to plot
    params = ['train/box_loss', 'train/cls_loss', 'train/dfl_loss',
              'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)',
              'metrics/mAP50-95(B)', 'val/box_loss', 'val/cls_loss', 'val/dfl_loss',
              'lr/pg0', 'lr/pg1', 'lr/pg2']
    
    # Assign colors and markers to each model
    model_styles = assign_styles(details_df['Model'].unique())
    
    # Plot individual plots
    if plot_mode in [0, 2]:
        plot_individual_plots(details_df, data_dir, plot_dir, model_styles, params, marker_size, line_width, legend_fontsize, dpi)
    
    # Collect and plot combined plots
    if plot_mode in [0, 1, 3]:
        plots_data, use_log_scale = collect_plot_data(details_df, data_dir, model_styles, params, log_scale_threshold)
        plot_combined_plots(plots_data, use_log_scale, plot_dir, line_width, legend_fontsize, dpi)

# Define data directory and plot output directory
data_dir = './'
plot_output_dir = './plots'

# Define selected train indices array
selected_train_indices = [12, 13, 14, 15, 17, 19, 27, 28, 29, 30, 31, 32, 36, 37, 38, 40, 41, 43, 51]

# Call function to generate plots
create_plots(data_dir, plot_output_dir, marker_size=0.5, line_width=0.5, legend_fontsize=5, dpi=1000, plot_mode=3, selected_train_indices=selected_train_indices)