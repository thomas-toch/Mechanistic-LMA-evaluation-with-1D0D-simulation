import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data aggregation from uploaded images
pathways = ['R_PCA-ACA', 'R_MCA-PCA', 'R_ACA-MCA', 'L_ACA-R_ACA', 'L_MCA-ACA', 'L_PCA-MCA', 'L_ACA-PCA']

data = {
    'Pathway': pathways * 4,
    'Case': ['CEA12']*7 + ['CEA13']*7 + ['ICSS2']*7 + ['ICSS3']*7,
    'Resistance': [
        1.99, 1.6, 14.33, 10.31, 2.55, 21.15, 4.1,      # CEA12
        4310.4, 2.26, 4.42, 0.84, 2.29, 19.9, 7.9,      # CEA13
        106.69, 32.77, 7.82, 5.45, 4.93, 7.96, 157.07,  # ICSS2
        1.69, 3.59, 2.06, 0.8, 1.84, 9.33, 1196.3       # ICSS3
    ],
    'n': [
        1.34, 14.26, 5.47, 3.28, 15.95, 4.28, 2.39,     # CEA12
        0.47, 12.79, 10.51, 13.05, 12.19, 5.92, 1.16,   # CEA13
        0.71, 5.05, 7.58, 7.35, 11.72, 8.22, 0.76,      # ICSS2
        3.35, 16.98, 19.95, 18.11, 30.01, 36.69, 0.9    # ICSS3
    ],
    'r_avg': [
        0.438, 0.214, 0.142, 0.188, 0.177, 0.135, 0.284, # CEA12
        0.048, 0.198, 0.169, 0.273, 0.2, 0.124, 0.29,    # CEA13
        0.144, 0.111, 0.156, 0.178, 0.157, 0.151, 0.123, # ICSS2
        0.341, 0.154, 0.176, 0.249, 0.160, 0.119, 0.059  # ICSS3
    ]
}

df = pd.DataFrame(data)

# Visualization
fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
plt.subplots_adjust(hspace=0.3)

# Panel 1: Log Resistance
sns.scatterplot(data=df, x='Pathway', y='Resistance', hue='Case', s=100, ax=axes[0], legend=False)
axes[0].set_yscale('log')
axes[0].set_ylabel('Resistance [mmHg*s/ml]')
axes[0].grid(True, which="both", ls="-", alpha=0.15)

# Panel 2: Number of Channels (n)
sns.scatterplot(data=df, x='Pathway', y='n', hue='Case', s=100, ax=axes[1], legend=False)
axes[1].set_ylabel('LMA Count (n)')
axes[1].grid(True, which='major', ls="-", alpha=0.15) # Main grid
axes[1].minorticks_on()                               # Turn on minor ticks
axes[1].grid(True, which='minor', ls=":", alpha=0.1)  # Finer, dotted minor grid

# Panel 3: Average Radius
sns.scatterplot(data=df, x='Pathway', y='r_avg', hue='Case', s=100, ax=axes[2], legend=False)
axes[2].set_ylabel('Avg. Radius [mm]')
axes[2].grid(True, which='major', ls="-", alpha=0.15)
axes[2].minorticks_on()
axes[2].grid(True, which='minor', ls=":", alpha=0.1)

plt.xticks(rotation=45)
plt.show()