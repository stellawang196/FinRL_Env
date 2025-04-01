import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib.dates as mdates
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from graphs.compute_stats import BackTestStats

# Load your CSV file
data = pd.read_csv("wandb_export_2024-07-25T14_42_18.664+02_00.csv")

# DJIA data, used for date range
djia = YahooDownloader(
    start_date="2021-01-01", end_date="2024-01-01", ticker_list=["DJIA"]
).fetch_data()


# Define the policies and colors for plotting
policies = [
    "DDPG",
    "SAC",
    "PPO",
    "Ensemble_10",
    "Ensemble_5",
    "Ensemble",
    "dow",
    "minvar",
]
policies2 = [
    "DDPG",
    "SAC",
    "PPO",
    "Ensemble 3",
    "Ensemble 2",
    "Ensemble 1",
    "DJIA",
    "Min Variance",
]
colors = ["blue", "green", "red", "purple", "orange", "pink", "black", "teal"]

# Set up a single plot
fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as necessary

# "time_start": "2021-01-01",
# "time_end": "2023-12-01",
start_date = datetime.strptime("2021-01-01", "%Y-%m-%d")
end_date = datetime.strptime("2023-12-01", "%Y-%m-%d")

# Create a date range that matches the length of the Step column
date_range = pd.date_range(start=start_date, end=end_date, periods=len(data))

assert len(date_range) == len(data["Step"])

policy_stats = []

for i, policy in enumerate(policies):
    # Extract data for each policy
    avg = data[f"{policy}_run - ensemble_reward"].values
    min_val = data[f"{policy}_run - ensemble_reward__MIN"].values
    max_val = data[f"{policy}_run - ensemble_reward__MAX"].values

    # Plot average rewards
    ax.plot(date_range, avg, label=f"{policies2[i]}", color=colors[i])

    df = pd.DataFrame({"daily_return": data[f"{policy}_run - ensemble_reward"]})
    df["date"] = djia["date"]

    policy_stats.append((policy, BackTestStats(df)))

# Set titles and labels
ax.set_title("Comparison of Cummulative Returns", fontsize=24)
ax.set_xlabel("Date", fontsize=22)
ax.set_ylabel("Cumulative Returns", fontsize=22)

ax.xaxis.set_major_locator(mdates.YearLocator())  # Change the interval as needed
ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%Y-%b")
)  # Change the format as needed

ax.tick_params(axis="both", which="major", labelsize=20)
ax.yaxis.get_offset_text().set_fontsize(20)
ax.legend(fontsize=16)
ax.grid(True)
plt.tight_layout()
plt.savefig("cum_reward_comparison.png")  # Saves as PNG file
plt.show()

print(policy_stats)
