#!/usr/bin/env python3


from numpy import log as ln
from numpy import trapz


def differential_entropy(pdf, x_pdf):           # pdf is a vector because we want to perform a numerical integration

    f = []                                      # Vector we want to integrate with the numerical integration function

    for i in range(0, len(pdf)):
        if pdf[i] > 0:                          # To avoid ln of 0 operation
            f.append(-1 * pdf[i] * ln(pdf[i]))
        else:
            f.append(0)

    ans = trapz(f, x_pdf)                       # Integrate using the composite trapezoidal rule
    return ans
