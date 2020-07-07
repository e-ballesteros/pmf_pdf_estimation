#!/usr/bin/env python3


import numpy as np

from differential_entropy import differential_entropy
from pdf_gaussian import gaussian_pdf
from sklearn.neighbors import KernelDensity

resolution = float(input('Introduce resolution of p.d.f. vector: '))
mean = float(input('Introduce mean of Gaussian distribution: '))
variance = float(input('Introduce variance of Gaussian distribution: '))

min_x = mean - 3 * variance                                                 # The width depends on variance
max_x = mean + 3 * variance

x = np.arange(min_x, max_x + resolution, resolution)                        # We want to include also de max_x

pdf = []                                                                    # Gaussian pdf

for i in range(0, len(x)):
    pdf.append(gaussian_pdf(x[i], mean, variance**(1/2)))                   # Computation of Gaussian pdf

dif_entropy = differential_entropy(pdf, x)                                  # Differential entropy of Gaussian dist.

print('Differential entropy of Gaussian distribution with resolution = ',
      resolution,
      ', mean = ',
      mean,
      ' and variance = ',
      variance,
      ' is ',
      dif_entropy)

pdf_disc = [i/sum(pdf) for i in pdf]                                        # Divide all elements in pdf by sum(pdf)

number_samples = int(input('Introduce number of samples: '))
samples = np.random.choice(x, p=pdf_disc, size=number_samples)              # Generation of samples of pdf of x

cont_samp_std = np.std(samples)                                             # Standard deviation of samples list
cont_samp_len = len(samples)                                                # Length of samples list
cont_samp_min = min(samples)                                                # Minimum value of samples list
cont_samp_max = max(samples)                                                # Highest value of samples list
margin = cont_samp_std * 2

kernelFunction = input('Introduce Kernel function you want to use (gaussian, tophat, epanechnikov,'
                       'exponential, linear or cosine): ')

optimal_bandwidth = 1.06 * cont_samp_std * np.power(cont_samp_len, -1/5)

user_optimal_bandwidth = input('Do you want to use optimal bandwidth (y/n) : ')

if user_optimal_bandwidth == 'y':
    bandwidthKDE = optimal_bandwidth
else:
    bandwidthKDE = float(input('Introduce bandwidth : '))


kde_object = KernelDensity(kernel=kernelFunction, bandwidth=bandwidthKDE).fit(samples.reshape(-1, 1))
X_plot = np.linspace(cont_samp_min - margin, cont_samp_max + margin, 1000)[:, np.newaxis]
kde_LogDensity_estimate = kde_object.score_samples(X_plot)
kde_estimate = np.exp(kde_LogDensity_estimate)

X_plot = np.linspace(cont_samp_min - margin, cont_samp_max + margin, 1000)  # x values of estimated pdf

dif_entropy_estimated = differential_entropy(kde_estimate, X_plot)

print('Differential entropy of estimated p.d.f from a set of samples of a Gaussian distribution with resolution = ',
      resolution,
      ', mean = ',
      mean,
      ' and variance = ',
      variance,
      ' is ',
      dif_entropy_estimated)

print('From a set of samples of a Gaussian distribution with resolution = ',
      resolution,
      ', mean = ',
      mean,
      ' and variance = ',
      variance,
      ',\nusing Kernel function: ',
      kernelFunction,
      ' with bandwidth = ',
      bandwidthKDE,
      ',\nthe relative difference (%) between the differential entropy of the p.d.f. and the estimated p.d.f. is: ',
      abs(dif_entropy_estimated - dif_entropy)/dif_entropy * 100,
      '%')

# Comentario de prueba
