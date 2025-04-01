import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")

    # Load your CSV file


data = pd.read_csv("crypto_samp.csv")
# data = pd.read_pickle("output/env_performance_data.pkl")
# Define the vector environments and colors for plotting
vecenvs = [2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
colors = plt.cm.tab20(np.linspace(0, 1, len(vecenvs)))

# Set up the plot
plt.figure(figsize=(8, 6))

window_size = 20  # Define the window size for smoothing

for i, vecenv in enumerate(vecenvs):

    avg = data[f"{vecenv} - reward_per_second"].values
    min_val = data[f"{vecenv} - reward_per_second__MIN"].values
    max_val = data[f"{vecenv} - reward_per_second__MAX"].values

    smoothed_average = moving_average(avg, 10)

    # Calculate mean difference between max and min
    mean_diff = (max_val - min_val).mean()
    # Calculate standard deviation
    std_dev = np.std(avg)

    # Plot the average and fill between min and max
    steps = data["Step"].values[: len(smoothed_average)]
    plt.plot(
        steps,
        smoothed_average,
        label=f"{vecenv}",
        color=colors[i],
        # alpha=0.7,
        linewidth=3,
    )
    # plt.fill_between(data["Step"], min_val, max_val, color=colors[i], alpha=0.3)

# Add titles and labels
plt.title("Samples per second for various env counts", fontsize=18)
plt.xlabel("Step", fontsize=18)
plt.ylabel("Samples per Second", fontsize=18)
plt.tick_params(axis="both", which="major", labelsize=14)

# Add legend
# plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc="upper left")
legend = plt.legend(
    fontsize=14,
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    title="Env count",
    frameon=False,
)

legend.get_title().set_fontsize("16")

# Add grid
plt.grid(True)

# Adjust layout to prevent overlap
# plt.tight_layout()
plt.tight_layout(
    rect=[0, 0, 1, 0.95]
)  # Adjust the bottom, top, left, and right margins

# Save the figure
plt.savefig("./output/crypto_samples.png")  # Saves as PNG file

# Show the plot
plt.show()
