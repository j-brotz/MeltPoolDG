#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 13:55:36 2021

@author: magdalena
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse


def curvature2(a, b, t):
    x, y = ellipse(a, b, t)
    return 1 / (a**2 * b**2 * (x**2 / a**4 + y**2 / b**4)**1.5)


def curvature(a, b, x, y):
    return 1 / (a**2 * b**2 * (x**2 / a**4 + y**2 / b**4)**1.5)


def ellipse(a, b, t):
    return (a * np.cos(t), b * np.sin(t))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Plot the curvature of an ellipse.')
    parser.add_argument('--radius', type=float, required=False, default=100.e-6,
                        help='define the radius of the ellipse to obtained the equivalent surface to a circle with the same radius.')
    parser.add_argument('--eccentricity', type=float, required=False, default=1.5,
                        help='define the eccentricity to determine the length of the semi-axes a=e*R and b=R/e')
    args = parser.parse_args()

    a_ = args.radius * args.eccentricity
    b_ = args.radius / args.eccentricity

    curv = []

    plt.close('all')

    angles = np.linspace(0, 2 * np.pi, 100)
    radii = []
    x_ = []
    y_ = []
    circle1_x = []
    circle1_y = []
    circle2_x = []
    circle2_y = []

    for t_ in angles:
        x, y = ellipse(a_, b_, t_)
        x_.append(x)
        y_.append(y)
        radii.append(np.sqrt(x**2 + y**2))
        curv.append(curvature(a_, b_, x, y))

    # ----------------------------------------------------------------------------
    # curvature of horizontal vertices
    # ----------------------------------------------------------------------------
    curv1 = 1 / (b_**2 / a_)
    e_2 = a_**2 - b_**2
    center_curv1 = [e_2 / a_, 0]
    for t_ in angles:
        x, y = ellipse(1 / curv1, 1 / curv1, t_)
        circle1_x.append(x + center_curv1[0])
        circle1_y.append(y + center_curv1[1])
    # ----------------------------------------------------------------------------
    # curvature of vertical vertices
    # ----------------------------------------------------------------------------
    curv2 = 1 / (a_**2 / b_)
    e_2 = a_**2 - b_**2
    center_curv = [0, e_2 / b_]
    for t_ in angles:
        x, y = ellipse(1 / curv2, 1 / curv2, t_)
        circle2_x.append(x + center_curv[0])
        circle2_y.append(y + center_curv[1])

    fig, axes = plt.subplots(1, 2)

    # ----------------------------------------------------------------------------
    # plot curvature and radius
    # ----------------------------------------------------------------------------
    ax = axes[0]
    ax2 = ax.twinx()
    ax2.plot(angles, curv, color="blue", label="curvature")
    ax2.tick_params(axis='y', colors='blue')
    ax2.set_ylabel("curvature")
    ax.set_ylabel("radius")
    ax.plot(angles, radii, color="r", label="radius")
    ax2.legend(loc="upper left")
    ax.set_xlabel("angle t")
    ax.set_xticks(np.arange(0, 2.01 * np.pi, np.pi / 4))
    labels = [r"{:}$\pi$".format(i) for i in np.arange(0, 2.01, 0.25)]
    ax.set_xticklabels(labels)

    # ----------------------------------------------------------------------------
    # plot ellipse and curvature circles at semi axes
    # ----------------------------------------------------------------------------
    ax = axes[1]
    ax.set_aspect('equal', 'box')
    ax.plot(x_, y_, label="ellipse", color="k")
    ax.plot(circle1_x, circle1_y, label="circle of curvature; k = {:.2f}".format(
        curvature2(a_, b_, 0)))
    ax.plot(circle2_x, circle2_y, label="circle of curvature; k = {:.2f}".format(
        curvature2(a_, b_, np.pi / 2)))
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    for ax in axes:
        ax.grid()
        ax.legend()

    print("curv semi axis: {:.2f} and {:.2f} ".format(
        curvature2(a_, b_, 0), curvature2(a_, b_, np.pi / 2)))

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

    plt.show()
