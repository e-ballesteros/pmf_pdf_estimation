#!/usr/bin/env python3


import numpy as np
from estimation_pmf import estimate_pmf
from entropy import entropy
from pdf_gaussian import gaussian_pdf
from sklearn.neighbors import KernelDensity

resolution = float(input('Introduce resolution of p.d.f. vector: '))
mean = float(input('Introduce mean of Gaussian distribution: '))
variance = float(input('Introduce variance of Gaussian distribution: '))

min_x = mean - 3 * variance                                                 # The width depends on variance
max_x = mean + 3 * variance

x = np.arange(min_x, max_x + resolution, resolution)                        # We want to include also de max_x
print('Vector x is: ', x)

# pdf = np.zeros(len(x))                                                      # Create pdf with same length as x
pdf = []

for i in range(0, len(x)):
    pdf.append(gaussian_pdf(x[i], mean, variance**(1/2)))

print('Discrete p.d.f. of Gaussian distribution is: ', pdf)
print('Summation of pdf is: ', sum(pdf))


# Opcion 1: poner p = pdf/sum(pdf)
number_samples = int(input('Introduce number of samples: '))
samples = np.random.choice(x, p=pdf, size=number_samples)                   # Generation of samples of pdf of x

cont_samp_std = np.std(samples)
cont_samp_len = len(samples)

optimal_bandwidth = 1.06 * cont_samp_std * np.power(cont_samp_len, -1/5)
bandwidthKDE = optimal_bandwidth
kernelFunction = 'gaussian'
kde_object = KernelDensity(kernel=kernelFunction, bandwidth=bandwidthKDE).fit(samples.reshape(-1, 1))

print('kde_object is: ', kde_object)


