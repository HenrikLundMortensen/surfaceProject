import numpy as np
import matplotlib.pyplot as plt

def plotFeatureVector(fVector):
    ''' Plot a single feature vector. First element of the array determines the oxygen density.
    Second element determines silver density and third element is the atomic number '''
    oDensity = fVector[0]
    agDensity = fVector[1]
    if fVector[2] == 8:
        plt.plot(oDensity,agDensity,'ro')
    if fVector[2] == 47:
        plt.plot(oDensity,agDensity,'go')

def plotFeatureMap(composite):
    ''' Plot a 3d  matrix consisting of feature vectors. The first dimension is the images. 
    The second dimension must contain the atoms. The third dimension is the feature vector.
    The feature vector must have first element as oxygen density, second element as silver density
    and third element the atomic number'''
    for i in range(0,composite.shape[0]):          # Loop over images
        for j in range(0,composite.shape[1]):      # Loop over atoms 
            if composite[i,j,2] == 8:              # Plot on oxygen plot
                plt.figure(1)
                plotFeatureVector(composite[i,j])
            if composite[i,j,2] == 47:              # Plot on siler plot
                plt.figure(2)
                plotFeatureVector(composite[i,j])

    plt.figure(1)
    plt.ylabel('Silver density')
    plt.xlabel('Oxygen density')
    plt.title('Oxygen atoms')

    plt.figure(2)
    plt.ylabel('Silver density')
    plt.xlabel('Oxygen density')
    plt.title('Silver atoms')
    plt.show()









