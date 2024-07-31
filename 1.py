import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

data_dir = './'
plot_output_dir = './plots'

# 定义要筛选的train_index 数组
selected_train_indices = [12, 13, 14, 15, 17, 19, 27, 28, 29, 30, 31, 32, 36, 37, 40, 41, 43, 51]

details_csv_path = os.path.join('runs/detect/train_details.csv')

    # 读取 train_details.csv 文件
details_df = pd.read_csv(details_csv_path)

details_df = details_df[details_df['Train Index'].isin(selected_train_indices)]

    # 打印筛选后的 details_df 以进行调试
print("Filtered details_df:")
print(details_df)