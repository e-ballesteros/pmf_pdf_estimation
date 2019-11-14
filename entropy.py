#!usr/bin/env python3


# Function that returns the entropy of a certain probability mass function (only one value is needed for binaries)
def entropy(prob):

    from numpy import log as ln

    # Binary random variable
    if len(prob) == 1:

        if prob[0] == 0 or prob[0] == 1:                                       # Preventing wrong operations
            ent = 0
        else:
            ent = -prob[0] * ln(prob[0]) - (1 - prob[0]) * ln(1 - prob[0])

    # Ternary random variable and larger sets
    else:
        ent = 0
        for i in range(0, len(prob)):
            if prob[i] > 0:                                                # Preventing wrong operations
                ent -= prob[i] * ln(prob[i])

    return ent
