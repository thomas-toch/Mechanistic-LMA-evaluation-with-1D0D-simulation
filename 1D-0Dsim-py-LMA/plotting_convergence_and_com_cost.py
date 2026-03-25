import pandas as pd
import matplotlib.pyplot as plt

# --- Configuration ---
# Add your file paths and legends here.
# Format: (Convergence_File, Time_File, Label)
file_pairs = [
    (
    r"C:\Users\To Chi Hang Thomas\PycharmProjects\1D-0D simulation\1D-0Dsim-py-LMA\output_20260228_080913_with_new_resistance_proportions\log_convergence_no_hot_start_prev_dt_dx.csv",
    r"C:\Users\To Chi Hang Thomas\PycharmProjects\1D-0D simulation\1D-0Dsim-py-LMA\output_20260228_080913_with_new_resistance_proportions\time_log_cycle.csv",
    'no hot start, prev dx, dt'),
    (
    r"C:\Users\To Chi Hang Thomas\PycharmProjects\1D-0D simulation\1D-0Dsim-py-LMA\output_20260228_095217_hot_start_for_new_resistance_proportions\log_convergence_hot_start_prev_dt_dx.csv",
    r"C:\Users\To Chi Hang Thomas\PycharmProjects\1D-0D simulation\1D-0Dsim-py-LMA\output_20260228_095217_hot_start_for_new_resistance_proportions\time_log_cycle.csv",
    'w/ hot start, prev dx, dt'),
    (r"C:\Users\To Chi Hang Thomas\PycharmProjects\1D-0D simulation\1D-0Dsim-py-LMA\output_20260301_204035_cea13_pred_no_hot_start_no_autoreg_improved_dxdt_vsurg\log_convergence_no_hot_start_large_dt_dx_truncated.csv",
r"C:\Users\To Chi Hang Thomas\PycharmProjects\1D-0D simulation\1D-0Dsim-py-LMA\output_20260301_204035_cea13_pred_no_hot_start_no_autoreg_improved_dxdt_vsurg\time_log_cycle_no_hot_start_large_dt_dx_truncated.csv",
     'no hot start, large dx, dt'),
    (r"C:\Users\To Chi Hang Thomas\PycharmProjects\1D-0D simulation\1D-0Dsim-py-LMA\output_20260301_212750_cea13_pred_yes_hot_start_no_autoreg_improved_dxdt_vsurg\log_convergence_hot_start_large_dt_dx_truncated_no_vsurg.csv",
r"C:\Users\To Chi Hang Thomas\PycharmProjects\1D-0D simulation\1D-0Dsim-py-LMA\output_20260301_212750_cea13_pred_yes_hot_start_no_autoreg_improved_dxdt_vsurg\time_log_cycle_hot_start_large_dt_dx_truncated_no_vsurg.csv",
     'w/ hot start, large dx, dt')
]


metrics = ['Q_max_rel_err', 'Parm_rel_err', 'Q_CoW_distal_max_rel_err']
time_cols = ['t_1d_lw [s]', 't_stn [s]', 't_0d [s]', 't_1d_bif [s]', 't_reg [s]', 't_out [s]']

# Global Aesthetic Adjustments
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10
})


def generate_plots():
    # # --- 1. CONVERGENCE VS CYCLES ---
    # for metric in metrics:
    #     plt.figure(figsize=(10, 6))
    #     # Add convergence threshold line
    #     plt.axhline(y=0.001, color='black', linestyle='--', linewidth=1.5, label='Convergence Threshold (0.001)')
    #
    #     for conv_path, _, label in file_pairs:
    #         df_conv = pd.read_csv(conv_path)
    #         if metric in df_conv.columns:
    #             plt.plot(df_conv['cycle no.'], df_conv[metric],
    #                      label=label, linewidth=2.5, marker='o', markersize=5)
    #
    #     plt.yscale('log')
    #     plt.xlabel('Number of Cycles')
    #     plt.ylabel(f'{metric} (Log Scale)')
    #     plt.title(f'Convergence: {metric} vs Cycles')
    #     plt.legend(loc='upper right', frameon=True)
    #     plt.grid(True, which="both", ls="-", alpha=0.2)
    #     plt.tight_layout()
    #     plt.show()

    # --- 2. PIE CHARTS ---
    # Using a dimmer, professional color palette
    pie_colors = plt.cm.Pastel1.colors

    for _, time_path, label in file_pairs:
        df_time = pd.read_csv(time_path)
        existing_cols = [c for c in time_cols if c in df_time.columns]
        time_sums = df_time[existing_cols].sum()

        plt.figure(figsize=(8, 8))
        # wedgeprops creates the white edges between slices
        plt.pie(time_sums, labels=existing_cols, autopct='%1.1f%%',
                startangle=140, colors=pie_colors,
                wedgeprops={'linewidth': 2, 'edgecolor': 'white'})

        plt.title(f'Time Distribution: {label}', pad=20)
        plt.tight_layout()
        plt.show()

    # --- 3. CONVERGENCE VS TOTAL TIME ---
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.axhline(y=0.001, color='black', linestyle='--', linewidth=1.5, label='Convergence Threshold (0.001)')

        for conv_path, time_path, label in file_pairs:
            df_conv = pd.read_csv(conv_path)
            df_time = pd.read_csv(time_path)
            df_merged = pd.merge(df_conv, df_time[['cycle no.', 't_total [s]']], on='cycle no.')

            if metric in df_merged.columns:
                plt.plot(df_merged['t_total [s]'], df_merged[metric],
                         label=label, linewidth=2.5, marker='o', markersize=5)

        plt.yscale('log')
        plt.xlabel('Total Time [s]')
        plt.ylabel('Residual (Log Scale)')
        plt.title(f'Convergence: {metric} vs Total Time')
        plt.legend(loc='upper right', frameon=True)
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    generate_plots()