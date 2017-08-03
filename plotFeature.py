import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

def plotFeatureVector(fVector):
    ''' Plot a single feature vector. First element of the array determines the atom type.
    0 = oxygen, 1 = silver. Second element is the oxygen density and third element is the silver density'''
    oDensity = fVector[1]
    agDensity = fVector[2]
    if fVector[0] == 0:
        plt.plot(oDensity,agDensity,'ro')
    if fVector[0] == 1:
        plt.plot(oDensity,agDensity,'go')

def plotFeatureMap(composite):
    ''' Plot a 3d  matrix consisting of feature vectors. The first dimension is the images. 
    The second dimension must contain the atoms. The third dimension is the feature vector.
    The feature vector must have the first element determine the atom type, the second element 
    determines the oxygen density and the third element is the silver density'''
    for i in range(0,composite.shape[0]):          # Loop over images
        for j in range(0,composite.shape[1]):      # Loop over atoms 
            if composite[i,j,0] == 0:              # Plot on oxygen plot
                plt.figure(1)
                plotFeatureVector(composite[i,j])
            if composite[i,j,0] == 1:              # Plot on siler plot
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


if __name__ == '__main__':
    composite = np.arange(75).reshape(5,5,3)
    for i in range(0,composite.shape[0]):
        for j in range(0,composite.shape[1]):
            composite[i,j,0] = np.random.randint(2)
            composite[i,j,1] = np.random.randint(10)
            composite[i,j,2] = np.random.randint(10)
    plotFeatureMap(composite)

