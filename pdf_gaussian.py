#!/usr/bin/env python3

from math import pi
from math import exp


# It returns the probability given a certain value of x and the mu and sigma of the gaussian distribution
def gaussian_pdf(x, mu, sigma):
    return 1.0 / (sigma * (2.0 * pi)**(1/2)) * exp(-1.0 * (x - mu)**2 / (2.0 * (sigma**2)))