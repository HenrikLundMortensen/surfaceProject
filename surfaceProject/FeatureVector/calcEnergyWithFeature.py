import numpy as np
from surfaceProject.FeatureVector.featureVector import *
#from plotGrid import *

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
    
def EBondFeature(f):
    """
    Calculates the energy contribution from a single feature vector. The rules are completely                                        made up.                                                                                                                                      
    Input:                                                                                                                           f: Featurevector ( [O density, Ag density, Atomic number,O-O Lenght,Ag-Ag Length,Ag-O Length] )                                              
    Output:                                                                                                                          E: Energy contribution of a single feature vector
    """
    E = 0

    if f[2] == 8:
        E -= 4*np.exp(-(f[0] - 0.4)**2/(0.2**2))
        E -= 4*np.exp(-(f[4] - 1.36)**2/(0.2**2))
        E -= 4*np.exp(-(f[3]-0)**2/(0.2**2))
        E -= 4*np.exp(-(f[5] - 1.09)**2/(0.2**2))
    if f[2] == 47:
        E -= 2*f[1]
        E -= 2*np.exp(-(f[0] - 0.8)**2/(0.2**2))
        E -= np.exp(-(f[3] - 0.14)**2/(0.8**2)) + np.exp(-(f[3] - 0.37)**2/(0.8**2))
        E -= np.exp(-(f[4]-0.37)**2/(0.8**2)) + np.exp(-(f[4]-1.60)**2/(0.8**2))
        E -= np.exp(-(f[5] - 1.09)**2/(0.8**2)) + np.exp(-(f[5] -1.71)**2/(0.8**2))
    return E

def EBondFeatureGrid(g):
    """                                                                                                                           
    Calculate the energy given a grid. The energy is calculated with our own rules based                                             on the feature vectors.
    Input:
    g: numpy array grid

    Output:
    E: sum of energy contributions from all feature vectors
    """
    E = 0
    for f in getBondFeatureVectors([g])[0]:
        E += EBondFeature(f)
    return E
