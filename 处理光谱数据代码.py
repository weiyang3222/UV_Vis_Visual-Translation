# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 22:01:45 2024

@author: Administrator
"""

# 打开文件
with open(r'D:\zb\statistic\光谱\20240326\1_OCV_LSF_STO_20240312#1_Transmission__0__20-25-31-715.txt') as file:
   # 跳过前14行
    for _ in range(14):
        next(file)
    
    # 初始化计数器
    line_count = 0
    
    # 读取剩下的行并打印
    for line in file:
        print(line.strip())  # 可以根据需要调整是否去掉行尾的换行符
        line_count += 1

# 打印总共的行数
print("Total lines:", line_count)
