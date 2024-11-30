# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib as mpl
import matplotlib.cm as cm
from scipy import integrate
import math
from pylab import mpl
from scipy.interpolate import interp1d
import datetime
from datetime import timedelta
import time

start_time = time.time()

mpl.rcParams.update({'font.size': 18})
mpl.rcParams['figure.figsize'] = [8, 6]

# path_1 = "E:\\Westlake University\\Data_Processing\\Birnessite_Electrochem\\20230412\\CV_2nd"

# filelist_1 = os.listdir(path_1)

colors = [cm.viridis(i) for i in np.linspace(0, 1, 20)]
colors2 = [cm.Accent(i) for i in np.linspace(0, 1, 8)]
colors3 = [cm.tab20b(i) for i in np.linspace(0, 1, 20)]
colors4 = [cm.Set3(i) for i in np.linspace(0, 1, 8)]
colors5 = [cm.Dark2(i) for i in np.linspace(0, 1, 8)]
colors6 = [cm.tab20c(i) for i in np.linspace(0, 1, 20)]
colors7 = [cm.coolwarm(i) for i in np.linspace(0, 1, 8)]
colors8 = [cm.PuBu(i) for i in np.linspace(0, 1, 20)]


############### 确定特定波段的索引值，无需改动##################
# df_1 = pd.read_csv(r'D:\Westlake University\OneDrive_List\OneDrive - 西湖大学\Westlake University\Data\Ocean_Optics\NNO Protonation\20240202\1_OCV_20240115_NNO_002_Transmission__0__14-50-24-183.txt', 
#                     skiprows=14, sep="\t", header = None, encoding = 'gbk')

df_1 = pd.read_csv(r'D:\zb\statistic\光谱\20240326\1_OCV_LSF_STO_20240312#1_Transmission__0__20-25-31-715.txt', 
                    skiprows=14, sep="\t", header = None, encoding = 'gbk')


UV_1 = (df_1.iloc[:, 2:]).values

wavelength_value = [250.197, 300.19, 350.683, 400.097, 449.999, 500.374, 550.438, 600.19, 650.386, 700.257, 749.802, 800.504, 850.118, 900.124, 950.504]

wavelength_label = ['250', '300', '350', '400', '450', '500', '550', '600', '650', '700', '750', '800', '850', '900', '950']

# index_250nm = []
index_250nm = []

index_lt = []

colors9 = [cm.coolwarm(i) for i in np.linspace(0, 1, len(wavelength_value))]

# index_250nm = []
# for j in range(0, len(UV_1[0,:])):
    
#     if UV_1[0,j] == 300.19:   ### 250.197, 300.19, 400.097, 430.542, 449.999, 500.374, 550.438, 600.19, 700.257, 800.504, 900.124, 950.504
        
#         index_250nm.append(j)
        
#         break

for j in range(0, len(UV_1[0,:])):
    
    if UV_1[0,j] == 250.197:   ### 350.683, 400.097, 430.542, 449.999, 450.777, 500.374, 550.438, 600.19, 650.386, 700.257, 749.802, 800.504, 850.118, 900.124
        
        index_250nm.append(j)
        
        break

for i in range(0, len(wavelength_value)):
    
    idx = np.where(UV_1[0,:] == wavelength_value[i])
    
    index_lt.append(int(idx[0]))



# for j in range(0, len(UV_1[0,:])):
    
#     if UV_1[0,j] == 800.504:   ### 400.097, 430.542, 449.999, 500.374, 550.438, 600.19, 700.257, 800.504, 900.124
        
#         index_250nm.append(j)
        
#         break

########################################################






############# Negative ###############

############ Current & OCV ############
# 假设文件夹路径为 folder_path

# # folder_path = 'E:/Westlake University/Data_Processing/NNO_Differentiation/20240323_Pulse-Exp/Data/1_tp_100ms'
# folder_path = 'E:/Westlake University/Data_Processing/NNO_Differentiation/20240323_Pulse-Exp/Data/3_tp_50ms_2nd'

# # 读取文件夹中的所有 CSV 文件，并将它们存储在一个列表中
# all_files = [file for file in os.listdir(folder_path) if file.endswith('.txt')]

# # 逐个读取 CSV 文件并将它们存储在一个 DataFrame 列表中
# dfs = []
# for file in all_files:
#     file_path = os.path.join(folder_path, file)
#     df = pd.read_csv(file_path, sep="\t")
#     dfs.append(df)

# # 使用 concat() 函数将 DataFrame 列表拼接成一个 DataFrame
# combined_df = pd.concat(dfs, ignore_index=True)

# # data_1 = pd.read_csv(r'E:\Westlake University\Data_Processing\NNO_Differentiation\20240323_Pulse-Exp\2_Pulse-Exp_100ms-10Pulse_20240115_NNO_004_Oxidized_C02.txt', sep="\t")
# data_1 = pd.read_csv(r'E:\Westlake University\Data_Processing\NNO_Differentiation\20240323_Pulse-Exp\5_Pulse-Exp_50ms-10Pulse_2nd_20240304_NNO_007_Oxidized_C02.txt', sep="\t")

############ UV-Vis ##############
 # df_2 = pd.read_csv(r'D:\Westlake University\OneDrive_List\OneDrive - 西湖大学\Westlake University\Data\Ocean_Optics\NNO Protonation\20240202\Used\1_UV-VIS_CA_-08V_Protonation_with_OCV.txt', 
#                     skiprows=14, sep="\t", header = None, chunksize=20000, encoding = 'gbk')


###############################

for j in range(0, len(wavelength_value)):
    
    # df_2 = pd.read_csv(r'E:\Westlake University\Data\Ocean_Optics\NNO Protonation\20240323\2_Pulse-Exp_100ms_20240115_NNO_004_Oxidized_Transmission__0__16-29-08-552.txt', 
    #                     skiprows=14, sep="\t", header = None, chunksize=20000, encoding = 'gbk')
    df_2 = pd.read_csv(r'D:\zb\statistic\光谱\20241006\2_LSF_STO_20240928_SPFC_Transmission__0__10-55-36-200.txt', 
                        skiprows=14, sep="\t", header = None, chunksize=20000, encoding = 'gbk')
    
    # if j <= 1:
    
    chunks_negative = []
    
    UV_negative = []
    
    UV_time_negative = []
    
    UV_total_negative = []
    
    for chunk in df_2:
        
        Sub_UV_1 = (chunk.iloc[1:, index_lt[j] + 0]).values
        Sub_UV_2 = (chunk.iloc[1:, index_lt[j] + 1]).values
        Sub_UV_3 = (chunk.iloc[1:, index_lt[j] + 2]).values
        Sub_UV_4 = (chunk.iloc[1:, index_lt[j] + 3]).values
        Sub_UV_5 = (chunk.iloc[1:, index_lt[j] + 4]).values
        
        Ave_Sub_UV = []
        
        for i in range(0, len(Sub_UV_1)):
            
            Ave = np.average(np.array([Sub_UV_1[i], Sub_UV_2[i], Sub_UV_3[i], Sub_UV_4[i], Sub_UV_5[i]]))
            
            Ave_Sub_UV.append(Ave)
        
    
        Sub_UV_time = (chunk.iloc[1:, 0]).values
        
        time_array = pd.to_datetime(Sub_UV_time)
        
        Delta_time_sec = [0]
        
        for i in range(1, len(time_array)):
            
            Delta_time_sec.append(time_array[i].timestamp()-time_array[i-1].timestamp())
             
        time_sec = []
        
        for i in range(0, len(Delta_time_sec)):
            
            sub_time = Delta_time_sec[3] * len(Delta_time_sec[:i])
            
            time_sec.append(sub_time + Delta_time_sec[3])   #### 此处Delta_time_sec[3]需要手动验证是否为正数，如果不是需要另外修改索引
            
        chunks_negative.append(chunk)
        
        UV_negative.append(Ave_Sub_UV)
        
        UV_time_negative.append(time_sec)
        
        
        UV_2_negative = (chunk.iloc[:, 2:]).values
        
        UV_total_negative.append(UV_2_negative)
        
        
    
    UV_array_negative = np.array([])
    
    
    
    Exp_time = np.linspace(0.2, 8100, 81000)
    
    # Exp_time = data_1['time/s'].values
    
    for i in range(0, len(UV_negative)):
        
        UV_array_negative = np.concatenate((UV_array_negative, np.array(UV_negative[i])))
        
        
    UV_total_array_negative = UV_total_negative[0]
     
    for i in range(1, len(UV_total_negative)):
        
        UV_total_array_negative = np.concatenate((UV_total_array_negative, np.array(UV_total_negative[i])), axis=0)


    Trans_data_negative = UV_total_array_negative[1:len(Exp_time)+1,:]

    Abs_UV_negative = -np.log(Trans_data_negative[:,index_lt[j]:index_lt[j]] * 0.01)


    UV_time_array_negative = np.linspace(min(Exp_time), max(Exp_time), len(UV_array_negative))

    Interp_UV_negative = np.interp(Exp_time, UV_time_array_negative, UV_array_negative)

    Norm_Abs_negative = -np.log(Interp_UV_negative * 0.01)
    
    
    ############ Data save ############
    Abs_array_negative = np.array([Exp_time, Norm_Abs_negative])
    Abs_dataframe_negative = pd.DataFrame(Abs_array_negative.T, columns=['time/s', 'Absorbance/a.u.'])
    # Abs_dataframe_negative.to_csv(r'E:\Westlake University\Data_Processing\NNO_Differentiation\20230927\Optics\Extracted Abs\250 nm\UV_Absorbance_time_1st.csv', index = False)
    # Abs_dataframe_negative.to_csv(r'D:\Westlake University\OneDrive_List\OneDrive - 西湖大学\Westlake University\Data_Processing\NNO_Differentiation\20240202\1_CA_-08V_Protonation_with_OCV\Extracted UV\UV_Absorbance_time_-08V-30min_with_OCV_1h_250nm.csv', index = False)
    # Abs_dataframe_negative.to_csv(r'E:\Westlake University\Data_Processing\NNO_Differentiation\20240323_Pulse-Exp\Extracted UV\1_tp_100ms\UV_Absorbance_time_10-Pulse_4-cycles_Total_tp_100ms_{}nm.csv'.format(wavelength_label[j]), index = False)
    # Abs_dataframe_negative.to_csv(r'D:\zb\statistic\光谱\20240326\LSF_STO_20240312_CV_test.csv'.format(wavelength_label[j]), index = False)
    ######################################
    
    
    
    
     
    
    
    
    
    
    figure1 = plt.figure(figsize=(9, 6))
    # plt.plot(Exp_time, Norm_UV_250nm, linewidth = 2, label = 'Transmittance at 250 nm', color = colors3[1], alpha = 0.8)
    plt.scatter(Exp_time, Norm_Abs_negative, linewidth = 3, label = 'Absorbance at ' + wavelength_label[j] + ' nm', color = colors9[j], alpha = 0.8)
    
    plt.legend(frameon = False, fontsize = 18, loc = 0, ncol = 1)
    
    plt.xlabel('Time (s)', fontsize = 20)
    
    plt.ylabel('Absorbance (a.u.)', fontsize = 20)
    
    # plt.ylim(-3.95, -3.05)
    ax1 = plt.gca()

    ax1.spines['bottom'].set_linewidth(2)
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['top'].set_linewidth(2)
    ax1.spines['right'].set_linewidth(2)
    
    
    plt.tight_layout()
    
    # plt.savefig(r'D:\Westlake University\OneDrive_List\OneDrive - 西湖大学\Westlake University\Data_Processing\NNO_Differentiation\20240202\1_CA_-08V_Protonation_with_OCV\Figures\2_UV(250nm)-Time_Plot_CA-30min_with_OCV_1h_Fig1.png', dpi = 600, transparent = True)
    # plt.savefig(r'E:\Westlake University\Data_Processing\NNO_Differentiation\20240323_Pulse-Exp\Figures\4_UV({}nm)-Time_Plot_10-Pulse_4-cycles_Total_tp_100ms_Fig1.png'.format(wavelength_label[j]), dpi = 600, transparent = True)
    # plt.savefig(r'D:\zb\statistic\PEDOT_PSS\20240808\图片\光谱\1_LSF_STO_20240312_CV_test({}nm).png'.format(wavelength_label[j]), dpi = 600, transparent = True)
    
    plt.show()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(total_time)
#%%
    # 初始化变量
chunks_negative = []
UV_negative = []
UV_time_negative = []
UV_total_negative = []
# 查找645 nm和655 nm的索引
index_start = next(i for i, v in enumerate(wavelength_value) if v >= 645)
index_end = next(i for i, v in enumerate(wavelength_value) if v > 655)

# 读取文件
df_2 = pd.read_csv(r'D:\zb\statistic\PEDOT_PSS\20240808\UV-vis\20240808\1_CA_0V_5min_-02_-08V_20240806_PEPSS_#1_Transmission__0__20-12-31-173.txt', 
                    skiprows=14, sep="\t", header = None, chunksize=20000, encoding = 'gbk')

# 处理数据块
for chunk in df_2:
    # 提取645 nm到655 nm波长范围内的数据
    Sub_UV_data = [chunk.iloc[1:, i].values for i in range(index_start, index_end + 1)]
    
    # 转换为吸光度
    Abs_UV_data = [-np.log(Sub_UV * 0.01) for Sub_UV in Sub_UV_data]
    
    # 计算平均吸光度
    Ave_Sub_Abs = np.mean(Abs_UV_data, axis=0)
    
    # 提取时间并转换为时间差
    Sub_UV_time = (chunk.iloc[1:, 0]).values
    time_array = pd.to_datetime(Sub_UV_time)
    
    Delta_time_sec = [0] + [time_array[i].timestamp() - time_array[i - 1].timestamp() for i in range(1, len(time_array))]
    time_sec = [Delta_time_sec[3] * len(Delta_time_sec[:i]) + Delta_time_sec[3] for i in range(len(Delta_time_sec))]
    
    chunks_negative.append(chunk)
    UV_negative.append(Ave_Sub_Abs)
    UV_time_negative.append(time_sec)
    UV_total_negative.append((chunk.iloc[:, 2:]).values)

# 整合数据
UV_array_negative = np.concatenate([np.array(UV_negative[i]) for i in range(len(UV_negative))])

# 实验时间
Exp_time = np.linspace(0.1, 8100, 81000)

# 插值并平滑处理
# Interp_UV_negative = np.interp(Exp_time, np.linspace(min(Exp_time), max(Exp_time), len(UV_array_negative)), UV_array_negative)

# 保存数据到txt文件
Abs_array_negative = np.array([Exp_time, Interp_UV_negative])
Abs_dataframe_negative = pd.DataFrame(Abs_array_negative.T, columns=['time/s', 'Absorbance/a.u.'])
# Abs_dataframe_negative.to_csv(r'D:\zb\statistic\PEDOT_PSS\20240808\650nm.txt', index=False, sep='\t')

# 打印操作完成信息
print("数据已保存为645-655 nm波长范围内吸光度与时间关系的txt文件。")
# 绘制吸光度与时间的关系图
plt.figure(figsize=(10, 6))
plt.plot(Abs_dataframe_negative['time/s'], Abs_dataframe_negative['Absorbance/a.u.'], label='650nm Absorbance')
plt.xlabel('Time (s)')
plt.ylabel('Absorbance (a.u.)')
plt.title('Absorbance vs. Time at 650nm')
plt.legend()
plt.grid(False)
plt.show()

