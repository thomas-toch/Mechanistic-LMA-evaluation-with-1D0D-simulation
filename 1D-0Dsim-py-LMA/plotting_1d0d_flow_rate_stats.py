import matplotlib.pyplot as plt
import numpy as np

# Case identifiers and Regions
cases = ['CEA12', 'CEA13', 'ICSS2', 'ICSS3']
regions = ['Rt. ACA', 'Lt. ACA', 'Rt. MCA', 'Lt. MCA', 'Rt. PCA', 'Lt. PCA']

# Data Mapping (Organized for plotting: SPECT, With LMA, Without LMA)
# Note: Reordered regions to match the image: R-ACA, L-ACA, R-MCA, L-MCA, R-PCA, L-PCA
data = {
    'CEA12': {
        'SPECT': [55.1, 53.8, 117.2, 108.7, 49.1, 50.2],
        'With_LMA': [60.2, 59.1, 127.8, 119.6, 53.5, 54.9],
        'Without_LMA': [61.2, 60.5, 130.4, 120.4, 52.8, 53.9]
    },
    'CEA13': {
        'SPECT': [54.2, 56.9, 115.2, 115.6, 43.0, 42.9],
        'With_LMA': [55.3, 57.9, 118.4, 117.2, 44.2, 43.7],
        'Without_LMA': [55.7, 57.8, 120.6, 115.1, 44.2, 43.9]
    },
    'ICSS2': {
        'SPECT': [90.8, 93.0, 254.2, 255.9, 93.1, 88.8],
        'With_LMA': [91.9, 94.0, 257.5, 258.0, 94.0, 89.3],
        'Without_LMA': [91.7, 93.9, 256.7, 257.6, 93.5, 89.0]
    },
    'ICSS3': {
        'SPECT': [74.0, 63.0, 206.0, 154.0, 79.0, 71.0],
        'With_LMA': [75.7, 65.1, 208.8, 161.9, 80.3, 72.3],
        'Without_LMA': [75.1, 64.1, 205.5, 159.1, 79.3, 71.8]
    }
}

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()
x = np.arange(len(regions))
width = 0.25

for i, case in enumerate(cases):
    ax = axes[i]

    # Plotting the three sets of bars
    ax.bar(x - width, data[case]['SPECT'], width, label='SPECT', color='#2ecc71')
    ax.bar(x, data[case]['With_LMA'], width, label='With LMA', color='#3498db')
    ax.bar(x + width, data[case]['Without_LMA'], width, label='Without LMA', color='#e74c3c')

    ax.set_title(f'Flow Comparison: {case}', fontsize=14, fontweight='bold')
    ax.set_ylabel('Flow [ml/min]')
    ax.set_xticks(x)
    ax.set_xticklabels(regions)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()