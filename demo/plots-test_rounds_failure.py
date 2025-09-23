import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the CSV data
df = pd.read_csv("demo/results.csv")  # Adjust path if needed

# Get unique protocols and noise models
protocols = df['protocol'].unique()
noise_models = df['noise_model'].unique()

# Create output directory (optional)
output_dir = "plots_by_setting"
os.makedirs(output_dir, exist_ok=True)

# Iterate through all combinations of protocol and noise_model
for protocol in protocols:
    for noise in noise_models:
        # Filter data for this protocol and noise model
        subset = df[(df['protocol'] == protocol) & (df['noise_model'] == noise)]

        if subset.empty:
            continue  # Skip if there's no data for this combination

        # Group by parameter and compute average failed test rounds
        avg_failed = (
            subset
            .groupby('parameter')['n_failed_test_rounds']
            .mean()
            .reset_index()
            .sort_values(by='parameter')
        )

        # Plot
        plt.figure(figsize=(8, 5))
        plt.plot(avg_failed['parameter'], avg_failed['n_failed_test_rounds'], marker='o')
        plt.xlabel('Parameter')
        plt.ylabel('Average Failed Test Rounds')
        plt.title(f'Protocol: {protocol} | Noise Model: {noise}')
        plt.grid(True)
        plt.tight_layout()

        # Save plot to file
        filename = f"{protocol}_{noise}.png".replace(" ", "_")
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        plt.close()  # Close the figure after saving

print("✅ All plots saved in:", output_dir)
