#!/usr/bin/env python3

import numpy as np


def estimate_pmf(samples):
    N = len(samples)
    samples_list, pmf_vector = np.unique(samples, return_counts=True)
    return samples_list, pmf_vector/N