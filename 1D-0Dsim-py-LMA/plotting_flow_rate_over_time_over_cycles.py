import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load your data (assuming it's comma-separated)
# Replace 'your_data_file.dat' with your actual file name
path = r'C:\Users\To Chi Hang Thomas\PycharmProjects\1D-0D simulation\1D-0Dsim-py-LMA\output_20260325_143532\Q_1d.dat'
df = pd.read_csv(path)

# Configuration
T_period = 1  # Heart rate period in seconds
artery_name = '40 L. int. carotid I [ml/s]' # Choose any column name from your header
tolerance = 1e-5

# 1. Filter for rows where time is a multiple of the period (t = 0, 0.9, 1.8...)
# We use modulo to find the start of each cycle
cycle_starts = df[np.isclose(df.iloc[:, 0] % T_period, 0, atol=tolerance)]

# 2. Extract cycle numbers and values
cycles = np.arange(len(cycle_starts))
values = cycle_starts[artery_name].values

# 3. Plot the drift
plt.plot(cycles, values, marker='o', linestyle='-', color='tab:blue')
plt.title(f'Cycle-to-Cycle Drift: {artery_name}')
plt.xlabel('Cycle Number')
plt.ylabel('Flow Rate at t=0 [ml/s]')
plt.grid(True, which='both', linestyle='--', alpha=0.5)

plt.show()