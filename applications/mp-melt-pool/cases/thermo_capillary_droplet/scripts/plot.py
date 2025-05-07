#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Plot the droplet velocity over time!')
    parser.add_argument('--file', type=str, required=True,
                        help='define the name of the text file, where the data is stored')
    args = parser.parse_args()
    plt.close('all')

    data_file = os.path.join(os.getcwd(), args.file)

    data = pd.read_csv(data_file)

    time = pd.DataFrame(data, columns=['time']).to_numpy()[:, 0]
    time_normalized = pd.DataFrame(data, columns=['t/tr']).to_numpy()[:, 0]
    vel_normalized = pd.DataFrame(data, columns=['u/ur']).to_numpy()[:, 0]
    vel_max = pd.DataFrame(data, columns=['u_max/ur']).to_numpy()[:, 0]
    vel_average = pd.DataFrame(data, columns=['u_avg/ur']).to_numpy()[:, 0]
    center_position = pd.DataFrame(data, columns=['y_center']).to_numpy()[:, 0]

    # data from Ma (2013)
    analytical_t = np.asarray([
        0.10229276895943551,
        0.19753086419753085,
        0.2998236331569665,
        0.40035273368606694,
        0.5026455026455026,
        0.603174603174603,
        0.7019400352733685,
        0.9029982363315694,
        0.8007054673721339,
        1.0035273368606699,
        1.0987654320987652,
        1.202821869488536,
        1.3033509700176364,
        1.4003527336860668,
        1.5061728395061724,
        1.6014109347442678,
        1.7019400352733682,
        1.8042328042328037,
        1.9082892416225747,
    ])

    analytical_vel = np.asarray([
        0.05956937799043062,
        0.09449760765550239,
        0.11411483253588517,
        0.1258373205741627,
        0.13253588516746412,
        0.13660287081339714,
        0.1394736842105263,
        0.1411483253588517,
        0.14043062200956938,
        0.1409090909090909,
        0.14186602870813397,
        0.1416267942583732,
        0.14186602870813397,
        0.1416267942583732,
        0.1409090909090909,
        0.14066985645933014,
        0.1411483253588517,
        0.14138755980861245,
        0.13995215311004786,
    ])

    fig, axs = plt.subplots(2)

    ax = axs[0]
    ax.plot(time_normalized, vel_max, label='max')
    ax.plot(time_normalized, vel_average, label='average')
    ax.plot(time_normalized, vel_normalized, label='centroid')
    ax.scatter(analytical_t, analytical_vel,
               color='k', label='analytical solution')
    ax.legend()
    ax.grid()
    ax.set_ylabel("u/ur")
    ax.set_xlabel("t/tr")
    ax.set_ylim([0, 0.15])
    ax.set_xlim([0, 2])

    ax = axs[1]
    ax.plot(time_normalized, center_position)
    ax.grid()
    ax.set_ylabel(r"$y_{center}$")
    ax.set_xlabel("t/tr")
    ax.set_xlim([0, 2])

    end_idx = np.searchsorted(time_normalized, 2)
    centre_idx = np.searchsorted(time_normalized, 1)

    ax.scatter([time_normalized[centre_idx], time_normalized[end_idx]],
               [center_position[centre_idx], center_position[end_idx]])
    ax.plot([time_normalized[centre_idx], time_normalized[end_idx], time_normalized[end_idx]],
            [center_position[centre_idx], center_position[centre_idx], center_position[end_idx]], "k")
    vel = (center_position[end_idx] - center_position[centre_idx]) / \
        (time[end_idx] - time[centre_idx]) / 0.024
    ax.annotate("u/ur={:}".format(float(vel)),
                (time_normalized[centre_idx], center_position[centre_idx]))

    fig.tight_layout()
    fig.savefig(data_file[:-4] + ".png")
    fig.show()
