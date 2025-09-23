import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ----------------------------
# Config
# ----------------------------
csv_path = "demo/results.csv"  # update if needed
thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.4, 1.0]

# Output dir for figures
out_dir = "plots_by_setting_multi_threshold"
os.makedirs(out_dir, exist_ok=True)

# ----------------------------
# Load and prep
# ----------------------------
df = pd.read_csv(csv_path)

# Ensure numeric types
df['parameter'] = pd.to_numeric(df['parameter'], errors='coerce')
df['n_failed_test_rounds'] = pd.to_numeric(df['n_failed_test_rounds'], errors='coerce')
df['match'] = df['match'].astype(str)

# Unique settings
protocols = df['protocol'].dropna().unique()
noise_models = df['noise_model'].dropna().unique()

# ----------------------------
# Iterate per (protocol, noise_model) and overlay thresholds
# ----------------------------
for protocol in protocols:
    for noise in noise_models:
        subset_all = df[(df['protocol'] == protocol) & (df['noise_model'] == noise)].copy()
        if subset_all.empty:
            continue

        # Prepare figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        handles = []
        labels = []

        for thr in thresholds:
            subset = subset_all.copy()
            # Acceptance rule
            subset['accepted'] = subset['n_failed_test_rounds'] < (thr * 100)
            subset['accepted_mismatch'] = subset['accepted'] & (subset['match'] != '✓')

            # Aggregate by parameter
            agg = (
                subset
                .groupby('parameter', dropna=True)
                .agg(
                    total=('circuit_label', 'count'),
                    accepted=('accepted', 'sum'),
                    accepted_mismatch=('accepted_mismatch', 'sum'),
                )
                .reset_index()
                .sort_values('parameter')
            )

            # Mismatch rate among accepted
            agg['mismatch_rate_among_accepted'] = np.where(
                agg['accepted'] > 0,
                agg['accepted_mismatch'] / agg['accepted'],
                np.nan
            )

            # Plot accepted count (top)
            h1, = ax1.plot(
                agg['parameter'],
                agg['accepted'],
                marker='o',
                label=f"thr={thr:.2f}"
            )

            # Plot mismatch rate among accepted (bottom)
            ax2.plot(
                agg['parameter'],
                agg['mismatch_rate_among_accepted'],
                marker='o'
            )

            # Collect legend handle/label once (from top axis)
            handles.append(h1)
            labels.append(f"thr={thr:.2f}")

        # Styling
        ax1.set_ylabel('Accepted (count)')
        ax1.set_title(f'Protocol: {protocol} | Noise Model: {noise}')
        ax1.grid(True)

        ax2.set_xlabel('Parameter')
        ax2.set_ylabel('Mismatch Rate (among accepted)')
        ax2.set_ylim(0, 1)
        ax2.grid(True)

        # Single legend for both subplots (based on top axis)
        ax1.legend(handles, labels, title='Threshold')

        plt.tight_layout()

        # Save figure
        safe_name = f"{protocol}_{noise}_multi_threshold.png".replace(" ", "_")
        plt.savefig(os.path.join(out_dir, safe_name), dpi=150, bbox_inches='tight')
        plt.close()

print(f"✅ Done. Figures saved in: {out_dir}")
