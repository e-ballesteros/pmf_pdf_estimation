#!/usr/bin/env python3
#Author: Mauro De Sanctis, PhD, University of Rome "Tor Vergata"

import numpy as np

def pmf(samples):
    N = len(samples)
    samples_list, pmf_vector = np.unique(samples, return_counts=True)
    return samples_list, pmf_vector/N

