import calcenergy as ce
import random
import numpy as np

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
    correctEnergy = -48
    surface = ce.randSurface(size)
    E = ce.calculateEnergy(surface,size)
    iter = 0
    while True:        
        iter+=1
        surface_temp = np.copy(surface)
        shuffle(surface,4)
        E2 = ce.calculateEnergy(surface,size)
        E1 = ce.calculateEnergy(surface_temp,size)
        if E2 == correctEnergy:
            break
        dE = E2 - E1
        if dE > 0:                               # We have a worse new surface
            if np.exp(-dE) < np.random.random(): # If much worse, higher chance to disregard
                surface = np.copy(surface_temp)  # Disregard new surface
    return surface
