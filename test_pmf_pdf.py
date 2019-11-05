#!/usr/bin/env python3
#Author: Mauro De Sanctis, PhD, University of Rome "Tor Vergata"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from pmf_estimation import pmf

print("Test probability mass function and probability density function estimation")

vector_length = 1000        #The larger the vector length, the more accurate is the probability estimation
np.random.seed(1)           #Use to generate same data over different runs
std_1 = 5
std_2 = 3
mean_1 = 33
mean_2 = 19
discrete_list = [1, 2, 3, 4]
discrete_probabilities = [0.1, 0.3, 0.2, 0.4]
discrete_samples = np.random.choice(discrete_list, p=discrete_probabilities, size=vector_length)
a = np.random.normal(loc=mean_1, scale=std_1, size=vector_length)
b = np.random.normal(loc=mean_2, scale=std_2, size=vector_length)
continuous_samples = np.hstack((a, b))      #Bimodal pdf if mean_1 is not equal to mean_2
#continuous_samples = a
cont_samp_min = min(continuous_samples)
cont_samp_max = max(continuous_samples)
cont_samp_std = np.std(continuous_samples)
cont_samp_mean = np.mean(continuous_samples)
cont_samp_iqr = np.percentile(continuous_samples, 75, interpolation='midpoint') \
    - np.percentile(continuous_samples, 25, interpolation='midpoint') #intequartile range
margin = cont_samp_std * 2
cont_samp_len = len(continuous_samples)

################## pmf Estimation ######################################################################################
discrete_list_estimated, pmf_estimated = pmf(discrete_samples)
f1 = plt.figure(1)
plt.stem(discrete_list_estimated, pmf_estimated)
########################################################################################################################

################### pdf Estimation #####################################################################################
###################   Histogram Method   #########################
optimal_bin_width = 3.5 * cont_samp_std * np.power(cont_samp_len, -1/3) #Scott's rule
#optimal_bin_width = 2 * cont_samp_iqr * np.power(cont_samp_len, -1/3)  #Freedman-Diaconis rule
bin_width = optimal_bin_width
number_of_Bins = int((max(continuous_samples) - min(continuous_samples))/bin_width)
hist_vector, bin_edges = np.histogram(continuous_samples, bins=number_of_Bins)
widthPlots = 0.8 * (bin_edges[1] - bin_edges[0])
center_of_Bins = (bin_edges[:-1] + bin_edges[1:]) / 2
f2 = plt.figure(2)
plt.bar(center_of_Bins, hist_vector, align='center', width=widthPlots)
##################################################################

###################   Kernel Method   ############################
optimal_bandwidth = 1.06 * cont_samp_std * np.power(cont_samp_len, -1/5)
bandwidthKDE = optimal_bandwidth                #As the bandwidth increases, the estimated pdf goes from being too rough to too smooth
kernelFunction = 'gaussian'     #Valid kernel functions are: ‘gaussian’|’tophat’|’epanechnikov’|’exponential’|’linear’|’cosine’
kde_object = KernelDensity(kernel=kernelFunction, bandwidth=bandwidthKDE).fit(continuous_samples.reshape(-1, 1))
X_plot = np.linspace(cont_samp_min - margin, cont_samp_max + margin, 1000)[:, np.newaxis]
kde_LogDensity_estimate = kde_object.score_samples(X_plot)
kde_estimate = np.exp(kde_LogDensity_estimate)
f3 = plt.figure(3)
plt.plot(X_plot, kde_estimate, '-')
########################################################################################################################

plt.show()  #Write 'plt.show()' only once, after the last figure.


