import os
import re
import pandas as pd

def extract_metrics(content):
    """
    从test.txt的内容中提取所需的metrics和speed信息。
    """
    # 定义正则表达式模式
    patterns = {
        'fitness': r"'fitness': np\.float64\((\d+\.\d+)\)",
        'metrics/mAP50(B)': r"'metrics/mAP50\(B\)': np\.float64\((\d+\.\d+)\)",
        'metrics/mAP50-95(B)': r"'metrics/mAP50-95\(B\)': np\.float64\((\d+\.\d+)\)",
        'metrics/precision(B)': r"'metrics/precision\(B\)': np\.float64\((\d+\.\d+)\)",
        'metrics/recall(B)': r"'metrics/recall\(B\)': np\.float64\((\d+\.\d+)\)"
    }
    
    results = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            results[key] = float(match.group(1))
        else:
            results[key] = None  # 如果未找到匹配项，设为None
    
    # 提取speed信息
    speed_pattern = r"speed: \{'preprocess': ([\d\.]+), 'inference': ([\d\.]+), 'loss': ([\d\.]+), 'postprocess': ([\d\.]+)\}"
    speed_match = re.search(speed_pattern, content)
    if speed_match:
        results['preprocess'] = float(speed_match.group(1))
        results['inference'] = float(speed_match.group(2))
        results['loss'] = float(speed_match.group(3))
        results['postprocess'] = float(speed_match.group(4))
        results['speed'] = results['preprocess'] + results['inference'] + results['loss'] + results['postprocess']
    else:
        results['preprocess'] = results['inference'] = results['loss'] = results['postprocess'] = results['speed'] = None
    
    return results

def main():
    # 定义runs目录路径
    runs_dir = 'runs'
    
    # 初始化数据列表
    data = []
    
    # 遍历runs目录下的所有子文件夹
    for run_name in os.listdir(runs_dir):
        run_path = os.path.join(runs_dir, run_name)
        if os.path.isdir(run_path):
            # 解析数据集类型
            if 'sample' in run_name:
                dataset = 'sample'
            elif 'complete' in run_name:
                dataset = 'complete'
            else:
                dataset = 'unknown'  # 如果既不是sample也不是complete
            
            # 处理val_best和val_last
            for model_type in ['val_best', 'val_last']:
                test_txt_path = os.path.join(run_path, model_type, 'test.txt')
                if os.path.exists(test_txt_path):
                    with open(test_txt_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 提取metrics和speed信息
                    metrics = extract_metrics(content)
                    
                    # 添加到数据列表
                    data.append({
                        'name': run_name,
                        'dataset': dataset,
                        'model': model_type,
                        'fitness': metrics.get('fitness'),
                        'metrics/mAP50(B)': metrics.get('metrics/mAP50(B)'),
                        'metrics/mAP50-95(B)': metrics.get('metrics/mAP50-95(B)'),
                        'metrics/precision(B)': metrics.get('metrics/precision(B)'),
                        'metrics/recall(B)': metrics.get('metrics/recall(B)'),
                        'preprocess': metrics.get('preprocess'),
                        'inference': metrics.get('inference'),
                        'loss': metrics.get('loss'),
                        'postprocess': metrics.get('postprocess'),
                        'speed': metrics.get('speed')
                    })
                else:
                    print(f"警告: {test_txt_path} 不存在。")
    
    # 创建DataFrame
    df = pd.DataFrame(data, columns=[
        'name', 'dataset', 'model', 'fitness', 'metrics/mAP50(B)',
        'metrics/mAP50-95(B)', 'metrics/precision(B)', 'metrics/recall(B)',
        'preprocess', 'inference', 'loss', 'postprocess', 'speed'
    ])
    
    # 保存为CSV文件
    output_csv = 'runs/results.csv'
    df.to_csv(output_csv, index=False)
    print(f"训练结果已汇总到 {output_csv}")

if __name__ == "__main__":
    main()