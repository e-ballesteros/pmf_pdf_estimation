#!/usr/bin/env python3

from differential_entropy import differential_entropy


x_min = 0
x_max = 1


def f(x):
    if (x > x_min) & (x < x_max/2):
        return 0.5
    elif (x > x_max/2) & (x < x_max):
        return 1.5
    else:
        return 0


print(differential_entropy(f))
