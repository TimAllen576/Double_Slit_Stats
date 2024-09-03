"""Perform various statistical analyses on Y2s data."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sc
import scipy.stats as stats
from matplotlib.animation import PillowWriter

WAVELENGTH = 1240 * 10 ** -9 / (25 * 10 ** 3)
microns = 10 ** -6
pixel_size = 75 * microns
min_distance = -138.5 * pixel_size
max_distance = 138.5 * pixel_size

I0 = 1  # Incident photon fluence
Ib = 0  # Background fluence
# d = 3 * microns  # Width of each slit
d = 1.6 * microns
L = 36  # Source to slits distance
R = 110  # Slit to detector distance
# D = 6 * microns  # Separation between slits
D = 5.5 * microns
s = 40 * microns  # Size of light source along y-axis (slit height?)
E = 25 * 10 ** 3  # Photon energy


# max_x = 289
# max_y = 276


def plotter(data, theoretical):
    """Plots a given dataset"""
    x_values = np.arange(len(data))
    fig, ax = plt.subplots()
    ax.plot(x_values, data)
    ax.plot(x_values, theoretical)
    ax.set(xlabel="X_pixel", ylabel="Counts", title="Y2s Data")
    plt.grid(True)
    plt.savefig("Y2s_full.svg")
    plt.show()


def chi_squared_test(observed, theoretical):
    """Performs a chi-squared test on the data."""
    chi_squared, p = stats.chisquare(observed, theoretical, 0)
    # chi2 = np.sum((observed - theroetical) ** 2 / theroetical)
    # print(f"Chi-squared: {chi2}")
    print(f"Chi-squared: {chi_squared}")
    print(f"P-value: {p}")
    return chi_squared, p


# def kolmogorov_smirnov_test(data):
#
# def anderson_darling_test(observed, theoretical):


# def cramer_von_mises_test(data):
#
def theoretical_distribution_2s(number_points=10000, wavelength=WAVELENGTH):
    """Creates a theoretical distribution for the data."""
    # wavelength = 1240 * 10 ** -9 / E
    A = 2 * wavelength * R / d  # Single slit equivalent (envelope)
    a = wavelength * R / D  # Period of 2s interference
    t = s * D / (wavelength * L)
    V = np.sin(np.pi * t) / (np.pi * t)
    x = np.linspace(min_distance, max_distance,
                    int((max_distance - min_distance) / pixel_size))
    y_values = (2 * I0 * (np.divide(
        np.sin(2 * np.pi * x / A), (2 * np.pi * x / A), out=np.ones_like(x),
        where=x != 0)) ** 2 * (1 + V * np.cos(2 * np.pi * x / a)) + Ib)
    scale = np.sum(y_values)
    y_values = y_values * number_points / scale
    return y_values


# def theoretical_distribution_1s(number_points,wavelength=WAVELENGTH):
#     """Creates a theoretical distribution for the data."""
#     # wavelength = 1240 * 10 ** -9 / E
#     A = 2 * wavelength * R / d  # Single slit equivalent (envelope)
#     a = wavelength * R / D  # Period of 2s interference
#     t = s * D / (wavelength * L)
#     V = np.sin(np.pi * t) / (np.pi * t)
#     x = np.linspace(min_distance, max_distance,
#                     int((max_distance - min_distance) / pixel_size))
#     y_values = (2 * I0 * (np.divide(
#         np.sin(2 * np.pi * x / A), (2 * np.pi * x / A), out=np.ones_like(x),
#         where=x != 0)) ** 2 * (1 + V * np.cos(2 * np.pi * x / a)) + Ib)
#     scale = np.sum(y_values)
#     y_values = y_values * number_points / scale
#     return y_values


def multinomial_test(observed, theoretical):
    """Performs a multinomial test on the data."""
    probs = observed / np.sum(observed)
    params = theoretical / np.sum(theoretical)
    test_statistic = -2 * np.sum(observed * np.log(
        np.divide(params, probs, out=np.ones_like(probs), where=probs != 0)))
    print(f"Test statistic: {test_statistic}")


def random_test():
    """Test function to generate random gaussian data."""
    x = np.linspace(-2, 2, 10000)
    gauss = 1 / np.sqrt(2 * np.pi) * np.exp(-x ** 2 / 2)
    scale = 1000
    gauss *= scale
    gauss_rand = np.clip(gauss + (scale / 100) * (np.random.rand(10000) - 0.5),
                         0, None)
    gauss_rand *= np.sum(gauss) / np.sum(gauss_rand)
    plt.scatter(x, gauss)
    plt.scatter(x, gauss_rand)
    plt.savefig("Gaussian.png")
    plt.clf()
    return gauss, gauss_rand


def plotter_anim(data, theoretical, y_values):
    """Plots a given dataset"""
    blocksize = len(y_values) // 100
    x = np.arange(len(data))
    fig, ax = plt.subplots()
    writer = PillowWriter(fps=1)
    with writer.saving(fig, "Y2s2.gif", 100):
        # ax.plot(x, theoretical, label="Theoretical")
        ax.set(xlabel="X_pixel", ylabel="Counts", title="Y2s Data")
        ax.grid(True)
        ax.set_xlim(0, 300)
        ax.set_ylim(0, 400)
        ln, = ax.plot(np.zeros(blocksize), np.zeros(blocksize), animated=True)

        for block in range(0, len(y_values), blocksize):
            new_data = np.bincount(y_values[block:block + blocksize])
            pad_length = len(data) - len(new_data)
            new_data = np.pad(new_data, (0, pad_length))
            ln.set_data(x, new_data)
            writer.grab_frame()


def masker(data, min_x, max_x, min_y, max_y):
    """Mask out bad data"""
    mask_x = (data["x"] >= min_x) & (data["x"] <= max_x)
    mask_y = (data["y"] >= min_y) & (data["y"] <= max_y)
    mask = mask_x & mask_y
    data_masked = data[mask]
    return data_masked


def calc_snr(obs_data, theory_1s, theory_2s):
    """Calculates a custom version of an SNR^2"""
    split_snr_1s = np.sum(
        2 * (obs_data - theory_1s) ** 2 / (obs_data + theory_1s))
    split_snr_2s = np.sum(
        2 * (obs_data - theory_2s) ** 2 / (obs_data + theory_2s))
    return split_snr_1s, split_snr_2s


def check_snr2(obs_values):
    """Check the SNR^2 for the 1s and 2s data"""
    theory_1s_values = np.loadtxt("Data/theory_1s.txt") * 1.1766
    min_y = 121
    num_y = 42
    max_y = min_y + num_y
    small_mask_obs = masker(obs_values, 126, 162, min_y, max_y)
    snr2_1s_list = []
    snr2_2s_list = []
    for num_frames in range(10, 1000):
        start_frame = 0
        y_values = small_mask_obs["y"][start_frame:start_frame + num_frames]
        small_mask_dist = np.bincount(y_values)
        # drop zero values from small_mask_dist
        small_mask_dist_new = small_mask_dist[min_y + 1:]
        # Pad the end with zeroes so it is 42 long
        small_mask_dist_pad = np.pad(small_mask_dist_new,
                                     (0, num_y - len(small_mask_dist_new)))
        theoretical_dist = theoretical_distribution_2s()
        small_theory_dist = theoretical_dist[min_y:max_y]
        small_theory_dist = small_theory_dist * len(y_values) / np.sum(
            small_theory_dist)

        theory_1s_values_scaled = theory_1s_values * num_frames / 70
        snr2_1s, snr2_2s = calc_snr(small_mask_dist_pad,
                                    theory_1s_values_scaled,
                                    small_theory_dist)
        snr2_1s_list.append(snr2_1s)
        snr2_2s_list.append(snr2_2s)
    # plt.plot(snr2_1s_list, label="1s")
    # plt.plot(snr2_2s_list, label="2s")
    # plt.title("SNR^2 for increasing photons in the 37x42 region")
    # plt.xlabel("Number of detected photons")
    # plt.ylabel("SNR^2")
    # plt.legend()
    # plt.show()
    xs = np.arange(len(small_mask_dist_pad))
    plt.plot(xs, small_mask_dist_pad, label="Experimental")
    plt.plot(xs, small_theory_dist, label="Theoretical 2s")
    plt.plot(xs, theory_1s_values, label="Theoretical 1s")
    plt.title("Sub-region plot of experimental and theoretical data")
    plt.ylabel("Counts")
    plt.xlabel("Y Pixel Coordinate")
    plt.legend()
    plt.show()
    # plt.savefig("Plots/Sub_region_plot.png")
    # print(f"Number of pixels: {len(y_values)}")
    # print(f"SNR^2 1s: {snr2_1s}")
    # print(f"SNR^2 2s: {snr2_2s}")
    return


def single_plotter(data, **settings):
    """Plots a given dataset"""
    x_values = np.arange(len(data))
    fig, ax = plt.subplots()
    ax.plot(x_values, data)
    ax.set(**settings)
    plt.grid(True)
    plt.show()


def extract_background(obs_values):
    """create the array bad area which is the number of pixels with x value
    below 50 for each y value"""
    bad_area = np.zeros(277)
    for i in range(277):
        bad_area[i] = np.sum(obs_values["x"][obs_values["y"] == i] < 50)
    bad_total = 2 * bad_area * (150 ** 2) / (50 ** 2)  # Extrapolation
    y_values = np.bincount(obs_values["y"])
    good_y = y_values - bad_total
    x_values = np.arange(len(good_y))
    fig, ax = plt.subplots()
    ax.plot(x_values, good_y)
    ax.set(xlabel="X_pixel", ylabel="Counts", title="Band subtracted data")
    plt.grid(True)
    plt.show()


def edge_plotter(data):
    """Plots the edge data"""
    edge_data = masker(data, 250, np.max(data["x"]), 0, np.max(
        data["y"]))
    y_values = np.bincount(edge_data["y"])
    # print the index of the maximum value
    print(np.argmax(y_values))
    fig, ax = plt.subplots()
    ax.plot(y_values)
    ax.set(xlabel="Y_pixel", ylabel="Counts", title="Edge data")
    plt.grid(True)
    plt.show()


def gaussian(x, a, b, c, offset):
    """Gaussian function"""
    return a * np.exp(-(x - b) ** 2 / c) + offset


def band_isolation(data):
    """Masks out the good and bad data then takes the difference"""
    bad_data = data[
        (data["y"] >= 115) & (data["y"] <= 125)]
    bad_x = np.bincount(bad_data["x"])
    # x_no_gap = np.concatenate((bad_x[:50], bad_x[250:]))
    # popt, _ = sc.optimize.curve_fit(gaussian, x_no_gap, np.ones_like(x_no_gap),
    #                     p0=[1, np.mean(x_no_gap), np.std(x_no_gap), 0])
    #
    # #




    good_data = data[
        (data["y"] >= 151) & (data["y"] <= 161)]
    good_x = np.bincount(good_data["x"])
    difference = bad_x - good_x
    single_plotter(difference, xlabel="X_pixel", ylabel="Counts",
                   title="Data subtracted band")


def main():
    """Loads and calls analyses on the data.
    original image size = 290 x 277 uint32-> 16-bit
    therefore approx midpoint = 138.5
    pixel real size = 75 microns"""
    obs_values = pd.read_csv("Data/full_data_masked.csv",
                             delimiter="\t", usecols=[0, 1],
                             dtype=int)

    check_snr2(obs_values)
    # extract_background(obs_values)
    # edge_plotter(obs_values)
    # band_isolation(obs_values)

    # plotter_anim(experimental_dist, theoretical_dist, y_values)
    # test_exact, test_random = random_test()
    # multinomial_test(experimental_dist, theoretical_dist)
    # chi_squared_test(test_random, test_exact)
    # chi_squared_test(experimental_dist, theoretical_dist)
    # plotter(experimental_dist, theoretical_dist)


if __name__ == "__main__":
    main()
