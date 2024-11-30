# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 22:38:09 2023

@author: Rolen
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
from concurrent.futures import ThreadPoolExecutor
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from multiprocessing import Pool, cpu_count
from PIL import Image



start_time = time.time()

mpl.rcParams.update({'font.size': 18})
# mpl.rcParams['figure.figsize'] = [8, 6]
'''
# 假设文件夹路径为 folder_path
# folder_path = 'E:/Westlake University/Data_Processing/NNO_Differentiation/20240323_Pulse-Exp/Data/1_tp_100ms'
folder_path = r'D:\zb\statistic\光谱\20240326\1_OCV_LSF_STO_20240312#1_Transmission__0__20-25-31-715.txt'

# 读取文件夹中的所有 CSV 文件，并将它们存储在一个列表中
all_files = [file for file in os.listdir(folder_path) if file.endswith('.txt')]

# 逐个读取 CSV 文件并将它们存储在一个 DataFrame 列表中
dfs = []
for file in all_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path, sep="\t")
    dfs.append(df)

# 使用 concat() 函数将 DataFrame 列表拼接成一个 DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

'''
# data_1 = pd.read_csv(r'E:\Westlake University\Data_Processing\NNO_Differentiation\20240323_Pulse-Exp\2_Pulse-Exp_100ms-10Pulse_20240115_NNO_004_Oxidized_C02.txt', sep="\t")
# data_1 = pd.read_csv(r'E:\Westlake University\Data_Processing\NNO_Differentiation\20240323_Pulse-Exp\5_Pulse-Exp_50ms-10Pulse_2nd_20240304_NNO_007_Oxidized_C02.txt', sep="\t")

############### 确定特定波段的索引值，无需改动##################

df_1 = pd.read_csv(r'D:\zb\statistic\光谱\20240326\1_OCV_LSF_STO_20240312#1_Transmission__0__20-25-31-715.txt', 
                    skiprows=14, sep="\t", header = None, encoding = 'gbk')

UV_1 = (df_1.iloc[:, 2:]).values


index_400nm = []
index_700nm = []
index_950nm = []
for j in range(0, len(UV_1[0,:])):
    
    if UV_1[0,j] == 400.097:   ### 250.197, 300.19, 400.097, 430.542, 449.999, 500.374, 550.438, 600.19, 700.257, 800.504, 900.124, 950.504
        
        index_400nm.append(j)
        
        break

for j in range(0, len(UV_1[0,:])):
    
    if UV_1[0,j] == 700.257:   ### 350.683, 400.097, 430.542, 449.999, 500.374, 550.438, 600.19, 700.257, 800.504, 900.124
        
        index_700nm.append(j)
        
        break
    
for j in range(0, len(UV_1[0,:])):
    
    if UV_1[0,j] == 950.504:   ### 400.097, 430.542, 449.999, 500.374, 550.438, 600.19, 700.257, 800.504, 900.124, 950.504
        
        index_950nm.append(j)
        
        break

########################################################

# df_2 = pd.read_csv(r'E:\Westlake University\Data\Ocean_Optics\NNO Protonation\20230927\2_CA_UV_20230714_NNO_1st_Transmission__0__14-48-18-317.txt', 
#                     skiprows=14, sep="\t", header = None, chunksize=20000)

df_2 = pd.read_csv(r'D:\zb\statistic\光谱\20241006\2_LSF_STO_20240928_SPFC_Transmission__0__10-55-36-200.txt', 
                    skiprows=14, sep="\t", header = None, chunksize=20000, encoding = 'gbk')


chunks = []

UV_700nm = []

UV_time = []

UV_total = []

for chunk in df_2:
    
    Sub_UV_700nm_1 = (chunk.iloc[1:, index_700nm[0] + 0]).values
    Sub_UV_700nm_2 = (chunk.iloc[1:, index_700nm[0] + 1]).values
    Sub_UV_700nm_3 = (chunk.iloc[1:, index_700nm[0] + 2]).values
    Sub_UV_700nm_4 = (chunk.iloc[1:, index_700nm[0] + 3]).values
    Sub_UV_700nm_5 = (chunk.iloc[1:, index_700nm[0] + 4]).values
    
    Ave_Sub_UV = []
    
    for i in range(0, len(Sub_UV_700nm_1)):
        
        Ave = np.average(np.array([Sub_UV_700nm_1[i], Sub_UV_700nm_2[i], Sub_UV_700nm_3[i], Sub_UV_700nm_4[i], Sub_UV_700nm_5[i]]))
        
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
        
    chunks.append(chunk)
    
    UV_700nm.append(Ave_Sub_UV)
    
    UV_time.append(time_sec)
    
    
    UV_2 = (chunk.iloc[:, 2:]).values
    
    UV_total.append(UV_2)
    
    

UV_700nm_array = np.array([])


Exp_time = np.linspace(0.1, 8100, 81000)


print(len(Exp_time))
#%%
for i in range(0, len(UV_700nm)):
    
    UV_700nm_array = np.concatenate((UV_700nm_array, np.array(UV_700nm[i])))
    
    
UV_total_array = UV_total[0]
 
for i in range(1, len(UV_total)):
    
    UV_total_array = np.concatenate((UV_total_array, np.array(UV_total[i])), axis=0)


Trans_data = UV_total_array[1:len(Exp_time)+1,:]

Abs_UV = -np.log(Trans_data[:,index_400nm[0]:index_950nm[0]] * 0.01)

Abs_UV = Abs_UV - Abs_UV[0]
    
Wavelength = UV_total_array[0,index_400nm[0]:index_950nm[0]]

# Abs_UV = Abs_UV[:295]

 #<=2160
#%%
level_min, level_max = np.min(Abs_UV), np.max(Abs_UV)
levels = np.linspace(level_min, level_max, 10000)
    
t = np.linspace(0.1, max(Exp_time), len(Abs_UV))

# t_plot = t[:600]

figure3 = plt.figure(figsize=(12, 10))
# num_plots = len(t)
# plot_width = 0.8 / num_plots
color_min, color_max = np.min(t), np.max(t)

colors = [cm.RdBu(i) for i in np.linspace(0, 1, len(t))]

for i in range (0, len(t)):
    plt.plot(Wavelength, Abs_UV[i], color = colors[i])

cmap1_name = 'RdBu'
    
cm1 = LinearSegmentedColormap.from_list(
    cmap1_name, colors, N=len(t))



norm = mpl.colors.Normalize(vmin=min(t), vmax=max(t))

sm1 = plt.cm.ScalarMappable(cmap=cm1, norm=norm)


sm1.set_array([])  

# plt.legend(frameon = False, fontsize = 15, bbox_to_anchor=(1.2, 0.5), loc = 'center', ncol = 1)
cb2 = plt.colorbar(sm1, ticks=np.linspace(int(min(t)), int(max(t)), 6)) 

cb2.set_label('Time (s)', fontsize=24) 
cb2.ax.tick_params(labelsize=24)

plt.xlabel('Wavelength (nm)', fontsize = 24)
plt.ylabel('Absorbance (a.u.)', fontsize = 24)

plt.xlim(400, 950)

plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

ax3 = plt.gca()
        
ax3.spines['bottom'].set_linewidth(2)
ax3.spines['left'].set_linewidth(2)
ax3.spines['top'].set_linewidth(2)
ax3.spines['right'].set_linewidth(2)

plt.tight_layout()


# plt.savefig(r'D:\zb\statistic\光谱\20240326\图片\LSF_STO_CV_test_20240312', dpi = 600, transparent = True)
# plt.savefig(r'D:\zb\statistic\PEDOT_PSS\20240808\图片\光谱\20240808_PEDOT_PSS_CA_0V_5min_mapping',dpi = 600, transparent = True)
#%%

#%%
  
figure3 = plt.figure(figsize=(12, 10))
t = np.linspace(0.1, max(Exp_time), len(Abs_UV))

# Abs_UV_Plot = Abs_UV[(int(t[0]*10)-1):(int(t[0]*10)-1)+len(t)]

level_min, level_max = np.min(Abs_UV), np.max(Abs_UV)
levels = np.linspace(level_min, level_max, 1000)
    
a=plt.contourf(Wavelength, t, Abs_UV, cmap='Spectral_r', levels = levels)
      
plt.xlim(400, 950)
plt.ylim(min(t), max(t))
plt.xlabel('Wavelength (nm)', fontsize = 24)
plt.ylabel('Time (s)', fontsize = 24)

# plt.title('E = ' + str(label_array[j]) + ', D = ' + str(D_show) + ' cm$^2$/s' + ', $\Lambda$ = ' + str(Lamda_show), fontsize = 18)

plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

cb = plt.colorbar(a,ticks = np.linspace(np.min(Abs_UV),np.max(Abs_UV),5))

cb.set_label('Absorbance (a.u.)', fontsize=24) 
cb.ax.tick_params(labelsize=24)

ax2 = plt.gca()

ax2.spines['bottom'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)
ax2.spines['top'].set_linewidth(2)
ax2.spines['right'].set_linewidth(2)
    # cb = plt.colorbar(a)
plt.tight_layout()

# plt.savefig(r'F:\OneDrive - 西湖大学\Westlake University\Data_Processing\NNO_Differentiation\20240202\1_CA_-08V_Protonation_with_OCV\Figures\2-UV_Contour_Plot_0-1800s_Fig1.png', dpi = 600, transparent = True)
# plt.savefig(r'D:\zb\statistic\光谱\20240326\图片\LSF_STO_CV_test_20240312_mapping', dpi = 600, transparent = True)


# plt.savefig(r'E:\Westlake University\Data_Processing\NNO_Differentiation\20230822\Optics\Figures\20230713_NNO_CA-UV_Negative_1st_CourtMAP_Z=Absorbance.png', dpi = 600, transparent = True)
# plt.savefig(r'E:\Westlake University\Data_Processing\NNO_Differentiation\20230822\Optics\Figures\20230713_NNO_CA-UV_Negative_2nd_CourtMAP_Z=Absorbance.png', dpi = 600, transparent = True)
# plt.savefig(r'E:\Westlake University\Data_Processing\NNO_Differentiation\20230822\Optics\Figures\20230713_NNO_CA-UV_Negative_3rd_CourtMAP_Z=Absorbance.png', dpi = 600, transparent = True)

# plt.savefig(r'E:\Westlake University\Data_Processing\NNO_Differentiation\20230822\Optics\Figures\20230713_NNO_CA-UV_Positive_1st_CourtMAP_Z=Absorbance.png', dpi = 600, transparent = True)
# plt.savefig(r'E:\Westlake University\Data_Processing\NNO_Differentiation\20230822\Optics\Figures\20230713_NNO_CA-UV_Positive_2nd_CourtMAP_Z=Absorbance.png', dpi = 600, transparent = True)
# plt.savefig(r'E:\Westlake University\Data_Processing\NNO_Differentiation\20230822\Optics\Figures\20230713_NNO_CA-UV_Positive_3rd_CourtMAP_Z=Absorbance.png', dpi = 600, transparent = True)
plt.savefig(r'D:\zb\statistic\PEDOT_PSS\20240808\图片\光谱\20240808_PEDOT_PSS_CA_0V_5min_mapping', dpi = 600, transparent = True)


# import matplotlib.pyplot as plt
# import numpy as np
# from concurrent.futures import ThreadPoolExecutor
#%%
# import matplotlib.pyplot as plt
# import numpy as np
# from concurrent.futures import ThreadPoolExecutor

def plot_contour(Abs_UV, Wavelength, t, levels):
    # figure2 = plt.figure(figsize=(12, 10))
    
    # a = plt.contourf(Wavelength, t, Abs_UV, cmap='RdBu', levels=levels)

    # plt.xlim(250, 950)
    # # plt.ylim(min(t), max(t))
    # plt.xlabel('Wavelength (nm)', fontsize=24)
    # plt.ylabel('Time (s)', fontsize=24)
    
    # # plt.yscale('log')
    
    # x_tick = [300, 400, 500, 600, 700, 800, 900]
    
    # plt.xticks(x_tick, fontsize=24)
    
    # plt.yticks(fontsize=24)

    # cb = plt.colorbar(a, ticks=np.linspace(np.min(Abs_UV), np.max(Abs_UV), 5))
    
    # cb.set_label('Absorbance (a.u.)', fontsize=24) 
    
    # cb.ax.tick_params(labelsize=24)

    # ax2 = plt.gca()
    # ax2.spines['bottom'].set_linewidth(2)
    # ax2.spines['left'].set_linewidth(2)
    # ax2.spines['top'].set_linewidth(2)
    # ax2.spines['right'].set_linewidth(2)

    # plt.tight_layout()

    # # Save the figure to a file
    # # plt.savefig(r'E:\Westlake University\Data_Processing\NNO_Differentiation\20230927\Optics\Figures\20230714_NNO_CA-UV_1st_Negative_CourtMAP_Z=Absorbance.png', dpi=600, transparent=True)
    # # plt.savefig(r'E:\Westlake University\Data_Processing\NNO_Differentiation\20230927\Optics\Figures\20230714_NNO_CA-UV_1st_Positive_CourtMAP_Z=Absorbance.png', dpi=600, transparent=True)
    # # plt.savefig(r'E:\Westlake University\Data_Processing\NNO_Differentiation\20230927\Optics\Figures\20230714_NNO_CA-UV_2nd_Negative_CourtMAP_Z=Absorbance.png', dpi=600, transparent=True)
    # plt.savefig(r'E:\Westlake University\Data_Processing\NNO_Differentiation\20231214\Figures\20231010_NNO_#2_CA-UV_-04V-1h_CourtMAP_Z=Absorbance.png', dpi=600, transparent=True)
    # # plt.savefig(r'E:\Westlake University\Data_Processing\NNO_Differentiation\20231120\Figures\20231010_NNO_#1_CA-UV_-08V-30min_350C-O2-2h_CA_-08V-30s_CourtMAP_Z=Absorbance_logt.png', dpi=600, transparent=True)

    # Create individual Abs_UV-Wavelength plots for different t values
    figure3 = plt.figure(figsize=(12, 10))
    # num_plots = len(t)
    # plot_width = 0.8 / num_plots
    color_min, color_max = np.min(t), np.max(t)
    
    colors = [cm.RdBu(i) for i in np.linspace(0, 1, len(t))]
    
    for i in range (0, len(t)):
        plt.plot(Wavelength, Abs_UV[i], color = colors[i])
    
    cmap1_name = 'RdBu'
        
    cm1 = LinearSegmentedColormap.from_list(
        cmap1_name, colors, N=len(t))
    
    
    
    norm = mpl.colors.Normalize(vmin=min(t), vmax=max(t))
    
    sm1 = plt.cm.ScalarMappable(cmap=cm1, norm=norm)
    
    
    sm1.set_array([])  
    
    # plt.legend(frameon = False, fontsize = 15, bbox_to_anchor=(1.2, 0.5), loc = 'center', ncol = 1)
    cb2 = plt.colorbar(sm1, ticks=np.linspace(int(min(t)), int(max(t)), 6)) 
    
    cb2.set_label('Time (s)', fontsize=24) 
    cb2.ax.tick_params(labelsize=24)
    
    plt.xlabel('Wavelength (nm)', fontsize = 24)
    plt.ylabel('Absorbance (a.u.)', fontsize = 24)
    
    plt.xlim(250, 900)
    
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    
    ax3 = plt.gca()
            
    ax3.spines['bottom'].set_linewidth(2)
    ax3.spines['left'].set_linewidth(2)
    ax3.spines['top'].set_linewidth(2)
    ax3.spines['right'].set_linewidth(2)
    
    plt.tight_layout()
    
    # Save the composite plot
    # plt.savefig(r'E:\Westlake University\Data_Processing\NNO_Differentiation\20240108\Figures\20231223_NNO_LAO_104_CA-UV_-08V-1h_Conductivity_Spectra_time-resolved_0-90s.png', dpi=600, transparent=True)
    
    
    # Create a 3D plot
    
#     figure4 = plt.figure(figsize=(18, 12))  # Create a new figure for the 3D plot
#     ax3d = figure4.add_subplot(111, projection='3d')
#     Wavelength_3d, t_3d = np.meshgrid(Wavelength, t)
#     ax3d.plot_surface(Wavelength_3d, t_3d, Abs_UV, cmap='RdBu')
#     ax3d.set_xlabel('Wavelength (nm)', fontsize=22)
#     ax3d.set_ylabel('Time (s)', fontsize=22)
#     ax3d.set_zlabel('Absorbance (a.u.)', fontsize=24)
    
#     # Adjust the position of axis labels and ticks to prevent overlap
#     ax3d.xaxis.labelpad = 20
#     ax3d.yaxis.labelpad = 15
#     ax3d.zaxis.labelpad = 11
    
#     # Adjust the position of the z-axis label and set its coordinates
#     # ax3d.zaxis.labelpad = 11
#     # ax3d.zaxis.set_label_coords(1, 1)  # Adjust the coordinates as needed
    
#     ax3d.set_zlabel('Absorbance (a.u.)', fontsize=20)
    
#     ax3d.view_init(elev=25, azim=65)
    
#     ax3d.invert_xaxis()
    
#     ax3d.xaxis.set_tick_params(labelsize=18)
#     ax3d.yaxis.set_tick_params(labelsize=18)
#     ax3d.zaxis.set_tick_params(labelsize=18)
    
#     # # plt.tight_layout()
    
# #     # Save the 3D plot
#     plt.savefig(r'E:\Westlake University\Data_Processing\NNO_Differentiation\20231214\Figures\20231010_NNO_#2_CA-UV_-04V-1h_Spectra_time-resolved_3D.png', dpi=600, transparent=True)
#     plt.close(figure4)

# # 使用 PIL 打开保存的图形并保存为最终图形
#     img = Image.open('temp_3D_Plot.png')
#     img.save(r'E:\Westlake University\Data_Processing\NNO_Differentiation\20231120\Figures\3D_Plot.png', dpi=(600, 600))

    
    # plt.close(figure4)
    
    # plt.savefig('111111.png', dpi = 600, transparent = True)
    # figure4.savefig('3D_Plot.png', dpi=600, transparent=True)
    # Show the 3D plot
    plt.show()


def main():
    # You need to define Abs_UV, Wavelength, and other variables before calling main()
    # Abs_UV_plot = Abs_UV[:97500]
    # Abs_UV_plot = Abs_UV
    
    # Wavelength_plot = Wavelength
    t = np.linspace(0.01, int(max(Exp_time)), len(Exp_time))
    # t = np.linspace(0, 10, 100)
    
    # t_i = t[:97500]
    # t_i = t[97500:]
    
    # Abs_UV_i = Abs_UV[:97500]
    # Abs_UV_i = Abs_UV[97500:]
    
    level_min, level_max = np.min(Abs_UV), np.max(Abs_UV)
    levels = np.linspace(level_min, level_max, 1000)

    # Create a ThreadPoolExecutor with desired number of threads
    num_threads = 12  # Adjust the number of threads as needed
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit the plot_contour task to the thread pool
        executor.submit(plot_contour, Abs_UV, Wavelength, t, levels)

    plt.show()

if __name__ == "__main__":
    main()





