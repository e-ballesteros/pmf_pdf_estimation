#!/usr/bin/env python3


import numpy as np
from estimation_pmf import estimate_pmf
from entropy import entropy
from pdf_gaussian import gaussian_pdf

resolution = float(input('Introduce resolution of p.d.f. vector: '))
mean = float(input('Introduce mean of Gaussian distribution: '))
variance = float(input('Introduce variance of Gaussian distribution: '))

# HAY QUE AJUSTAR ESTO
half_width = 10
min_x = mean - half_width
max_x = mean + half_width
# HAY QUE AJUSTAR ESTO

x = np.arange(min_x, max_x + resolution, resolution)                        # We want to include also de max_x
print('Vector x is: ', x)

pdf = np.zeros(len(x))                                                      # Create pdf with same length as x

for i in range(0, len(x)):
    pdf[i] = gaussian_pdf(x[i], mean, variance**(1/2))

print('Discrete p.d.f. of Gaussian distribution is: ', pdf)