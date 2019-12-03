#!/usr/bin/env python3


from numpy import inf                       # We will need to integrate with infinite limits
from numpy import log as ln
from numpy import trapz


def differential_entropy(pdf, x_pdf):       # pdf is a vector because we want to perform a numerical integration

    f = []

    for i in range(0, len(pdf)):
        if pdf[i] > 0:
            f.append(-1 * pdf[i] * ln(pdf[i]))
        else:
            f.append(0)

    ans = trapz(f, x_pdf)
    return ans
