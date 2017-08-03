import numpy as np
from featureVector import *
from plotGrid import *

def EFeature(f):
    """
    Calculates the energy contribution from a single feature vector. The rules are completely 
    made up. 

    Input:
    f: Featurevector ( [O density, Ag density, Atomic number] )

    Output:
    E: Energy contribution of a single feature vector
    """
    E = 0

    if f[2] == 8:
        E -= 4*np.exp(-(f[0] - 0.4)**2/(0.8**2))
    
    if f[2] == 47:
        E -= 2*f[1]
        E -= 2*np.exp(-(f[0] - 0.8)**2/(0.8**2))

    return E

    

def EFeatureGrid(g):
    """
    Calculate the energy given a grid. The energy is calculated with our own rules based
    on the feature vectors. 

    Input:
    g: numpy array grid

    Output:
    E: Sum of energy contributions from all feature vectors
    """
    E = 0
    for f in getFeatureVectors([g])[0]:
        E += EFeature(f)

    return E
    

