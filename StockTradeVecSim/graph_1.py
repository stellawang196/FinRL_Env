import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib.dates as mdates

def moving_average(data, window_size):
    data = np.asarray(data)
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")

# Load your CSV file
# data = pd.read_csv("wandb_gggg.csv")
data = pd.read_csv("wandb_export_2024-07-05T13_27_02.911+02_00.csv")

# Define the policies and colors for plotting
letters = ["a", "b", "c", "d"]
policies = ["DDPG", "SAC", "PPO", "simple ensemble"]
policies2 = ["DDPG", "SAC", "PPO", "Ensemble"]
colors = ["blue", "green", "red", "purple"]

# Set up a grid of plots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # Adjust the figure size as necessary
axs = axs.flatten()  # Flatten the array of axes to easily index them

start_date = datetime.strptime("2020-01-01", "%Y-%m-%d")
end_date = datetime.strptime("2023-01-01", "%Y-%m-%d")

date_range = pd.date_range(start=start_date, end=end_date, periods=len(data))

assert len(date_range) == len(data["Step"])

for i, policy in enumerate(policies):
    # avg = data[f"{policy}_run - reward_per_second"].values
    # min_val = data[f"{policy}_run - reward_per_second__MIN"].values
    # max_val = data[f"{policy}_run - reward_per_second__MAX"].values
    avg = data[f"{policy}_run - ensemble_reward"].values
    min_val = data[f"{policy}_run - ensemble_reward__MIN"].values
    max_val = data[f"{policy}_run - ensemble_reward__MAX"].values

    # Calculate mean, variance, and standard deviation
    combined_data = np.vstack((min_val, max_val))
    std_dev_between = np.std(combined_data, axis=0).mean()

    mean_diff = (max_val - min_val).mean()

    avg_smooth = moving_average(avg, 10)
    min_val_smooth = moving_average(min_val, 10)
    max_val_smooth = moving_average(max_val, 10)
    date_range_smooth = date_range[: len(avg_smooth)]

    # Plot on the respective subplot
    axs[i].plot(date_range_smooth, avg_smooth, label=f"Average", color=colors[i])
    axs[i].fill_between(date_range_smooth, min_val_smooth, max_val_smooth, color=colors[i], alpha=0.3)
    axs[i].set_title(
        f"{policies2[i]} ({letters[i]})", fontsize=25
    )  # Increase title font size
    axs[i].set_xlabel("Date", fontsize=23)  # Increase x-axis label font size
    axs[i].set_ylabel(
        "Cumulative Returns", fontsize=23
    )  # Increase y-axis label font size

    axs[i].tick_params(
        axis="both", which="major", labelsize=20
    )  # Adjust tick labels' font size
    axs[i].yaxis.get_offset_text().set_fontsize(20)
    
    axs[i].xaxis.set_major_locator(mdates.YearLocator())  # Change the interval as needed
    axs[i].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))  # Change the format as needed

    # Annotate statistics
    stats_text = f"Std dev: {std_dev_between:.2f}"
    axs[i].text(
        0.1,
        0.1,
        stats_text,
        transform=axs[i].transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontsize=23,  # Increase annotation font size
    )

    axs[i].legend(loc='upper right',fontsize=18)  # Increase legend font size
    axs[i].grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure
plt.savefig("reward_comparison.png")  # Saves as PNG file
# plt.savefig(
#     "reward_comparison.svg", format="svg"
# )  # Saves as SVG file for vector graphics quality
# plt.savefig("reward_comparison.pdf", format="pdf")  # Saves as PDF file

# Show the plot
plt.show()
