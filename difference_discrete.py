#!/usr/bin/env python3


import numpy as np
from estimation_pmf import estimate_pmf
from entropy import entropy

in_string = input('Introduce marginal p.m.f. of X separated by commas: ')
pmf = list(map(float, in_string.split(',')))                                # Split input by commas into list in float

number_samples = int(input('Introduce number of samples: '))

discrete_list = []                                                          # Domain of the random variable

for i in range(0, len(pmf)):
    discrete_list.append(i+1)

samples = np.random.choice(discrete_list, p=pmf, size=number_samples)       # Generation of samples
sample_vector, estimated_pmf = estimate_pmf(samples)                        # Estimation of p.d.f. giving the samples

ent_pmf = entropy(pmf)                                                      # Entropy of p.m.f. introduced by user
ent_estimated_pmf = entropy(estimated_pmf)                                  # Entropy of estimated p.m.f.

print('The entropy of the introduced p.m.f. is: ', ent_pmf)
print('The entropy of the estimated p.m.f. is: ', ent_estimated_pmf)
print('The difference between the introduced p.m.f. and the estimated p.m.f. is: ', abs(ent_pmf - ent_estimated_pmf))
