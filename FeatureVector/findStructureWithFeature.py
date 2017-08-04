from calcEnergyWithFeature import *
import random
import numpy as np

def randSurface(N): # Creates random surface matrix
    o = 3*N - 9
    ag = N*N - 3*N + 2
    surface = np.zeros(N*N)
    for i in range(0,ag):
        surface[i] = 1
    for i in range(ag,ag+o):
        surface[i] = 2
    surface = np.random.permutation(surface)
    surface = np.reshape(surface,(N,N))
    return surface

def shuffle(surface,M): # Shuffle M atoms on the surface                                                                          
    N = surface.shape[1]
    L = []
    x = 0
    while x < M:
        i = random.randint(0,N-1)
        j = random.randint(0,N-1)
        if (i,j) not in L:
            L.append((i,j))
            x += 1
    shuffleArray = np.empty([M,3])
    for i in range(M):
            shuffleArray[i,0] = surface[L[i][0],L[i][1]]
            shuffleArray[i,1] = L[i][0]
            shuffleArray[i,2] = L[i][1]
    np.random.shuffle(shuffleArray[:,0]) # Shuffle first column                                                                   
    for i in range(M):
        surface[L[i][0],L[i][1]] = shuffleArray[i,0]

def findOptimum(size):  # Find optimal structure                                                                                  
    surface = randSurface(size)
    iter = 0
    while iter < 500:
        iter+=1
        surface_temp = np.copy(surface)
        shuffle(surface,4)
        E2 = EBondFeatureGrid(surface)
        E1 = EBondFeatureGrid(surface_temp)
        dE = E2 - E1
        if dE > 0:                               # We have a worse new surface
            if np.exp(-dE) < np.random.random(): # If much worse, higher chance to disregard
                surface = np.copy(surface_temp)  # Disregard new surface
    return surface

if __name__ == '__main__':
#    import os,sys,inspect
#    import calcenergy as ce
#    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#    parentdir = os.path.dirname(currentdir)
#    sys.path.insert(0,parentdir)

 #   import calcenergy as ce
    import plotGrid as pg
    
    surface = findOptimum(5)
    print(surface)
#    print(pg.calculateEnergy(surface,5))
    fig = pg.initializePlotGridFigure(5)
    pg.plotGrid(surface,fig)
