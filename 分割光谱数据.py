# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:35:14 2024

@author: Administrator
"""

import pandas as pd
import numpy as np

def process_data(input_file, output_dir, time_col='time/s', absorbance_col='Absorbance/a.u.', time_interval=300):
    # 读取txt文件
    df = pd.read_csv(input_file, delimiter='\t')  # 这里假设数据是用制表符分隔的

    # 检查列名是否存在
    if time_col not in df.columns or absorbance_col not in df.columns:
        raise ValueError(f"文件中缺少必要的列: '{time_col}' 或 '{absorbance_col}'")

    # 获取时间和吸光度列
    time_data = df[time_col]
    absorbance_data = df[absorbance_col]

    # 将时间数据转换为浮点数（如果需要的话）
    time_data = time_data.astype(float)

    # 确定数据的时间范围
    start_time = time_data.min()
    end_time = time_data.max()

    # 切分数据并保存
    segment_start = start_time
    segment_end = segment_start + time_interval

    segment_index = 0
    while segment_start <= end_time:
        # 选择当前时间段的数据
        segment_mask = (time_data >= segment_start) & (time_data < segment_end)
        segment_time = time_data[segment_mask]
        segment_absorbance = absorbance_data[segment_mask]

        # 将时间调整为相对于当前段的时间
        segment_time_adjusted = segment_time - segment_start

        # 保存数据到文件
        segment_df = pd.DataFrame({
            time_col: segment_time_adjusted,
            absorbance_col: segment_absorbance
        })

        # 定义文件名并保存
        segment_file_name = f"{output_dir}/segment_{segment_index}.txt"
        segment_df.to_csv(segment_file_name, sep='\t', index=False)

        print(f"已保存数据到: {segment_file_name}")

        # 更新时间段
        segment_start += time_interval
        segment_end += time_interval
        segment_index += 1

# 调用函数，设置输入文件和输出目录
input_file = r'D:\zb\statistic\PEDOT_PSS\20240808\650nm.txt'  # 替换为实际的输入文件路径
output_dir = r'D:\zb\statistic\PEDOT_PSS\20240808\UV-vis\20240808\分割数据'  # 替换为实际的输出目录路径
process_data(input_file, output_dir)
