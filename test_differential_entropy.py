#!/usr/bin/env python3


import numpy as np

from differential_entropy import differential_entropy


x_min = 0
x_max = 1

resolution = 0.01                       # Resolution of pdf vector
num = (x_max - x_min)/resolution        # Number of elements of pdf vector


def f(x):
    if (x > x_min) & (x < x_max/2):
        return 0.5
    elif (x > x_max/2) & (x < x_max):
        return 1.5
    else:
        return 0


x_pdf = np.linspace(x_min, x_max, num)  # X values of pdf vector
pdf = []

for i in range(0, len(x_pdf)):
    pdf.append(f(x_pdf[i]))

print(differential_entropy(pdf, x_pdf))     # The pdf vector and its x values
