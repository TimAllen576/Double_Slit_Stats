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
MIN_DISTANCE = -138.5 * pixel_size
MAX_DISTANCE = 138.5 * pixel_size

I0 = 1.0  # Incident photon fluence
Ib = 0.0  # Background fluence
# d = 3 * microns  # Width of each slit
d_halfopt = 1.6 * microns
L = 36.0  # Source to slits distance
R = 110.0  # Slit to detector distance
# D = 6 * microns  # Separation between slits
D_halfopt = 5.5 * microns
s_unopt = 40 * microns  # Size of light source along y-axis (slit height?)
E = 25 * 10 ** 3  # Photon energy
POPT = [1.56251700e-04, 1.68774246e-06, 5.41761927e-06]
sub_region_photons = 207570


MAX_x = 289
MAX_y = 276


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
def theoretical_distribution_2s(s=s_unopt, d=d_halfopt,
        D=D_halfopt, number_points=10000, wavelength=WAVELENGTH):
    """Creates a theoretical distribution for the data."""
    # wavelength = 1240 * 10 ** -9 / E
    A = 2 * wavelength * R / d  # Single slit equivalent (envelope)
    a = wavelength * R / D  # Period of 2s interference
    t = s * D / (wavelength * L)
    V = np.sin(np.pi * t) / (np.pi * t)
    x = np.linspace(MIN_DISTANCE, MAX_DISTANCE,
                    int((MAX_DISTANCE - MIN_DISTANCE) / pixel_size))
    y_values = (2 * I0 * (np.divide(
        np.sin(2 * np.pi * x / A), (2 * np.pi * x / A), out=np.ones_like(x),
        where=x != 0)) ** 2 * (1 + V * np.cos(2 * np.pi * x / a)) + Ib)
    scale = np.sum(y_values)
    y_values = y_values * number_points / scale
    return y_values


def theoretical_distribution_opt(x, s=s_unopt, d=d_halfopt, D=D_halfopt):
    """Creates a theoretical distribution for the data."""
    A = 2 * WAVELENGTH * R / d  # Single slit equivalent (envelope)
    a = WAVELENGTH * R / D  # Period of 2s interference
    t = s * D / (WAVELENGTH * L)
    V = np.sin(np.pi * t) / (np.pi * t)
    y_values = (2 * I0 * (np.divide(np.sin(2 * np.pi * x / A),
        (2 * np.pi * x / A),
        out=np.ones_like(x),where=x != 0)) ** 2 * (
            1 + V * np.cos(2 * np.pi * x / a)) + Ib)
    y_values_scaled = 207570 * y_values / np.sum(y_values)
    return y_values_scaled


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
    gauss_rand = np.clip(gauss + (scale / 10) * (np.random.rand(10000) - 0.5),
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
    """Mask out bad data, includes mins but not maxes"""
    mask_x = (data["x"] >= min_x) & (data["x"] < max_x)
    mask_y = (data["y"] >= min_y) & (data["y"] < max_y)
    mask = mask_x & mask_y
    data_masked = data[mask]
    return data_masked


def calc_snr(obs_data, theory_1s, theory_2s):
    """Calculates a custom version of an SNR^2"""
    split_snr_1s = 2 * np.sum(
        (obs_data - theory_1s) ** 2 / (obs_data + theory_1s))
    split_snr_2s = 2 * np.sum(
        (obs_data - theory_2s) ** 2 / (obs_data + theory_2s))
    return split_snr_1s, split_snr_2s


def check_snr2(obs_values):
    """Check the SNR^2 for the 1s and 2s data"""
    theory_1s_values = np.loadtxt("Data/theory_1s.txt") * 1.1766
    min_y = 121
    num_y = 42
    max_y = min_y + num_y
    obs_subregion = masker(obs_values, 126, 162, min_y, max_y)
    snr2_1s_list = []
    snr2_2s_list = []
    x = np.linspace(MIN_DISTANCE, MAX_DISTANCE,
                    int((MAX_DISTANCE - MIN_DISTANCE) / pixel_size))
    theoretical_dist_2s = theoretical_distribution_opt(x, *POPT)
    subregion_2s = theoretical_dist_2s[min_y - 1:max_y - 1]
    for num_frames in range(10, 207570):
        start_frame = 0
        obs_frame_limit = obs_subregion["y"][start_frame:start_frame + num_frames]
        obs_dist_frame_limit = np.bincount(obs_frame_limit)
        # drop zero values from small_mask_dist
        obs_drop_zero = obs_dist_frame_limit[min_y:]
        # Pad the end with zeroes so it is 42 long
        obs_padded = np.pad(obs_drop_zero, (0, num_y - len(obs_drop_zero)))
        theory_2s_scaled = subregion_2s * len(obs_frame_limit) / np.sum(
            subregion_2s)
        theory_1s_values_scaled = theory_1s_values * len(obs_frame_limit) / 70
        # obs_padded = np.divide(obs_padded, np.sum(obs_padded))
        # theory_2s_scaled = np.divide(theory_2s_scaled, np.sum(theory_2s_scaled))
        # theory_1s_values_scaled = np.divide(theory_1s_values_scaled,
        #                                    np.sum(theory_1s_values_scaled))
        snr2_1s, snr2_2s = calc_snr(obs_padded,
                                    theory_1s_values_scaled,
                                    theory_2s_scaled)
        snr2_1s /= num_frames
        snr2_2s /= num_frames
        snr2_1s_list.append(snr2_1s)
        snr2_2s_list.append(snr2_2s)
    plt.plot(snr2_1s_list, label="1s")
    plt.plot(snr2_2s_list, label="2s")
    plt.title("SNR^2 for increasing photons in the 37x42 region")
    plt.xlabel("Number of detected photons")
    plt.ylabel("SNR^2")
    plt.legend()
    plt.show()
    # print(f"SNR^2 1s: {snr2_1s}")
    # print(f"SNR^2 2s: {snr2_2s}")
    xs = np.arange(len(obs_padded))
    plt.plot(xs, obs_padded, ".-", label="Experimental")
    plt.plot(xs, theory_2s_scaled, ".-", label="Theoretical 2s")
    plt.plot(xs, theory_1s_values_scaled, ".-", label="Theoretical 1s")
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


def gaussian(x, a, b, c, offset):
    """Gaussian function"""
    return a * np.exp(-(x - b) ** 2 / c) + offset


def band_isolation(data):
    """Masks out the good and bad data then takes the difference"""
    y_width = 1
    area_list = []
    for y_start in np.arange(0, MAX_y + 1, y_width):
        bad_data = data[
            (data["y"] >= y_start) & (data["y"] <= y_start + y_width)]
        bad_x = np.bincount(bad_data["x"])
        x = np.arange(len(bad_x))
        # Mask out the good data
        mask = (x < 50) | (x > 250)
        x_edges = x[mask].astype(float)
        data_edges = bad_x[mask].astype(float)
        for stddev in [10000000, 1000000, 100000, 50000, 22000]:
            try:
                # noinspection PyTupleAssignmentBalance
                popt, _ = sc.optimize.curve_fit(gaussian, x_edges, data_edges,
                                                p0=[np.max(bad_x), 150, stddev, 0])
                break
            except:
                continue
        data_central_pred = gaussian(x, *popt)
        area = np.trapezoid(data_central_pred, x)
        area_list.append(area)
    spread_x = np.repeat(area_list, y_width) // y_width
    final_section = y_width - (len(spread_x) - (MAX_y+1))
    if final_section == 0:
        bad_summed_x = spread_x
    else:
        bad_summed_x = spread_x[:MAX_y+1]
        bad_summed_x[len(bad_summed_x)-final_section:] *= y_width // final_section
    return bad_summed_x


def band_iso_single(data):
    """Masks out the good and bad data then takes the difference"""
    y_width = 1
    area_list = []
    y_start = 150
    bad_data = data[
        (data["y"] >= y_start) & (data["y"] <= y_start + y_width)]
    bad_x = np.bincount(bad_data["x"])
    x = np.arange(len(bad_x))
    # Mask out the good data
    mask = (x < 50) | (x > 250)
    x_edges = x[mask].astype(float)
    data_edges = bad_x[mask].astype(float)
    for stddev in [10000000, 1000000, 100000, 50000, 22000]:
        try:
            # noinspection PyTupleAssignmentBalance
            popt, _ = sc.optimize.curve_fit(gaussian, x_edges, data_edges,
                                            p0=[np.max(bad_x), 150, stddev, 0])
            break
        except:
            continue
    # print(popt)
    data_central_pred = gaussian(x, *popt)
    plt.plot(bad_x, ".-", label='Original Data')
    plt.plot(x, data_central_pred,
             label='Fitted Central Data')
    plt.legend()
    plt.show(block=True)
    difference = bad_x - data_central_pred
    single_plotter(difference, xlabel="X_pixel", ylabel="Counts",
                   title="Data subtracted band")


def subregion_optmized(data):
    min_y = 121
    num_y = 42
    max_y = min_y + num_y
    obs_subregion = masker(data, 126, 162, min_y, max_y)
    obs_dist = np.bincount(obs_subregion["y"])
    obs_drop_zero = obs_dist[min_y:]
    # Right shift one
    y_coords = np.arange(-18, 24, dtype=float) * pixel_size
    # popt, pcov = sc.optimize.curve_fit(
    #     theoretical_distribution_opt, y_coords, obs_drop_zero,
    #     p0=[s_unopt, d_halfopt, D_halfopt],
    #     bounds=([microns, 0.1*microns, 0.1*microns], [1, 1, 1]))
    popt, pcov = sc.optimize.curve_fit(
        theoretical_distribution_opt, y_coords, obs_drop_zero,
        p0=[s_unopt],
        bounds=([microns], [1]))
    theoretical_distribution = theoretical_distribution_opt(y_coords, *popt)
    plt.plot(y_coords/microns, obs_drop_zero, ".-", label="Experimental")
    plt.plot(y_coords/microns, theoretical_distribution, ".-", label="Theoretical")
    # plt.title(f"Optimized sub-region plot with \ns={popt[0]/microns:.2f} microns, "
    #           f"d={popt[1]/microns:.2f} microns, D={popt[2]/microns:.2f} microns")
    plt.title(f"Optimized sub-region plot with \ns={popt[0]/microns:.2f} microns, "
              f"d={d_halfopt/microns:.2f} microns, D={D_halfopt/microns:.2f} microns")
    plt.xlabel("Y dist from centre of interference(microns)")
    plt.ylabel("Counts")
    plt.legend()
    plt.show()


def main():
    """Loads and calls analyses on the data.
    original image size = 290 x 277 uint32-> 16-bit
    therefore approx midpoint = 138.5
    pixel real size = 75 microns"""
    obs_values = pd.read_csv("Data/full_data_masked.csv",
                             delimiter="\t", usecols=[0, 1],
                             dtype=int)
    data = obs_values.value_counts(sort=False)
    dz = data.values
    x = data.index.get_level_values(0)
    y = data.index.get_level_values(1)
    image_data = np.vstack((x, y, dz)).T
    np.savetxt("Data/image_data.csv", image_data, delimiter=",", fmt="%d")
    print(np.max(dz))
    # y_values = np.bincount(obs_values["y"])
    # x_values = np.arange(len(y_values), dtype=float)

    # z = np.zeros_like(x)
    # dx = np.ones_like(x)*0.5
    # dy = np.ones_like(x)*0.5
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # ax.bar3d(x, y, z, dx, dy, dz)
    # plt.show()

    # cut_145 = y_values[145:]
    # cut_half = y_values[139:]
    # theoretical_values_145 = theoretical_distribution_opt(x_values[145:], *POPT)
    # theory_145_scaled = theoretical_values_145 * len(y_values) / np.sum(
    #     theoretical_values_145)
    # subregion_optmized(obs_values)
    # check_snr2(obs_values)
    # band_iso_single(obs_values)
    # bad_summed_x = band_isolation(obs_values)
    # single_plotter(bad_summed_x, xlabel="Y_pixel", ylabel="Counts",
    #                  title="Bad data isolation")


    # plotter_anim(experimental_dist, theoretical_dist, y_values)
    # test_exact, test_random = random_test()
    # multinomial_test(experimental_dist, theoretical_dist)
    # chi_squared_test(test_random, test_exact)
    # multinomial_test(test_random, test_exact)
    # chi_squared_test(cut_145, theory_145_scaled)
    # plotter(experimental_dist, theoretical_dist)


if __name__ == "__main__":
    main()
