# -*- coding: utf-8 -*-
"""
Created on Thu May 16 11:41:08 2024

@author: Administrator
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load data from the TXT file
txt_data = pd.read_csv(r'D:\zb\statistic\电导\20240326\2_LSF_STO_20240312#1_CV_-1_1_test_03_CV_C01.txt', delimiter='\t')

# Extract required columns from TXT data
txt_time_current = txt_data[['time/s', '<I>/mA']]

# Load data from the CSV file
csv_data = pd.read_csv(r'D:\zb\statistic\光谱\20240326\LSF_STO_20240312_CV_test\各波长强度文件\LSF_STO_20240312_CV_test_450nm.csv')

# Extract required columns from CSV data
csv_time_absorbance = csv_data[['time/s', 'Absorbance/a.u.']]

# Create the plot
fig, ax1 = plt.subplots(figsize=(10, 8))

# Create a secondary y-axis
ax2 = ax1.twinx()

# Plot Absorbance data on the primary y-axis (left)
ax1.scatter(csv_time_absorbance['time/s'], csv_time_absorbance['Absorbance/a.u.'], color='#2A7EBA', label='Absorbance')
ax1.set_xlabel('Time (s)', fontsize=22)
ax1.set_ylabel('Absorbance (a.u.)', color='#2A7EBA', fontsize=22)
ax1.tick_params(axis='y', labelcolor='#2A7EBA', labelsize=20)
ax1.grid(False)  # Remove grid lines for ax1
# Plot Current data on the secondary y-axis (right)
ax2.plot(txt_time_current['time/s'], txt_time_current['<I>/mA'], color='#E58579', label='Current',linewidth=5)
ax2.set_ylabel('Current (mA)', color='#E58579', fontsize=22)
ax2.tick_params(axis='y', labelcolor='#E58579', labelsize=20)
ax2.grid(False)  # Remove grid lines for ax1
# Set x-axis limits for ax1 (primary axis)
ax1.set_ylim(0.15, 0.19)

# Add legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc=(0.05,0.85), fontsize=20,frameon=False)

plt.tight_layout()
plt.grid(False)
plt.show()
