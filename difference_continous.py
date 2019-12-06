#!/usr/bin/env python3


import numpy as np

from differential_entropy import differential_entropy
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

# pdf = np.zeros(len(x))                                                      # Create pdf with same length as x
pdf = []

for i in range(0, len(x)):
    pdf.append(gaussian_pdf(x[i], mean, variance**(1/2)))

print('Differential entropy of Gaussian distribution with resolution = ',
      resolution,
      ', mean = ',
      mean,
      ' and variance = ',
      variance,
      ' is ',
      differential_entropy(pdf, x))

pdf_disc = [i/sum(pdf) for i in pdf]                                           # Divide all elements by sum(pdf)

# Opcion 1: poner p = pdf/sum(pdf)
number_samples = int(input('Introduce number of samples: '))
samples = np.random.choice(x, p=pdf_disc, size=number_samples)                   # Generation of samples of pdf of x

cont_samp_std = np.std(samples)
cont_samp_len = len(samples)
cont_samp_min = min(samples)
cont_samp_max = max(samples)
margin = cont_samp_std * 2

optimal_bandwidth = 1.06 * cont_samp_std * np.power(cont_samp_len, -1/5)
bandwidthKDE = optimal_bandwidth
kernelFunction = 'gaussian'
kde_object = KernelDensity(kernel=kernelFunction, bandwidth=bandwidthKDE).fit(samples.reshape(-1, 1))
X_plot = np.linspace(cont_samp_min - margin, cont_samp_max + margin, 1000)[:, np.newaxis]
kde_LogDensity_estimate = kde_object.score_samples(X_plot)
kde_estimate = np.exp(kde_LogDensity_estimate)

X_plot = np.linspace(cont_samp_min - margin, cont_samp_max + margin, 1000)

print('Differential entropy of estimated p.d.f from a set of samples of a Gaussian distribution with resolution = ',
      resolution,
      ', mean = ',
      mean,
      ' and variance = ',
      variance,
      ' is ',
      differential_entropy(kde_estimate, X_plot))
