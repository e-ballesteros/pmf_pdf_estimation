#!/usr/bin/env python3

from scipy.integrate import quad
from numpy import inf                       # We will need to integrate with infinite limits
from numpy import log as ln


def differential_entropy(function):         # Function must be a function declared outside differential_entropy.py

    def f(x):
        return -function * ln(function)

    ans, err = quad(f, -inf, +inf)   # Integration of function with infinite range
