"""Code to take saved xy coords, remove bad pixels slices with multiple
data points,  save and plot"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plotter(data):
    """Bin and plot the data"""
    x = data["x"]
    y = data["y"]
    xs = np.bincount(x)
    ys = np.bincount(y)
    tx = np.arange(len(xs))
    ty = np.arange(len(ys))
    fig, ax1 = plt.subplots()

    color = "tab:red"
    ax1.set_ylabel("Count")
    ax1.set_xlabel("Y Pixel Coordinate", color=color)
    ax1.plot(ty, ys, color=color)
    ax1.tick_params(axis="x", labelcolor=color)

    ax2 = ax1.twiny()  # instantiate a second axes that shares the same y-axis

    color = "tab:blue"
    ax2.set_xlabel("X Pixel Coordinate",
                   color=color)  # we already handled the x-label with ax1
    ax2.plot(tx, xs, color=color)
    ax2.tick_params(axis="x", labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig("xy profiles.png")
    plt.show()


def main():
    """Main function"""
    path = "SAMPLE_Y00_Z00.txt"
    data = pd.read_csv(
        path, names=["x", "y", "slice", "value"], delimiter="\t")
    mask_x = data["x"] == 150
    mask_y = data["y"] == 73
    mask = mask_x & mask_y
    data = data[~mask]
    data.drop_duplicates(subset="slice", inplace=True, keep=False)
    # data.to_csv("cleaned_data.csv", index=False)
    plotter(data)


if __name__ == '__main__':
    main()
