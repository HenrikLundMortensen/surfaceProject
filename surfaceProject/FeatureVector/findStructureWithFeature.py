from surfaceProject.FeatureVector.calcEnergyWithFeature import *
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
    E2 = 0
    E1 = 0
    while iter < 50000:
        iter+=1
        surface_temp = np.copy(surface)
        shuffle(surface,4)
        E2 = EBondFeatureGrid(surface)
        E1 = EBondFeatureGrid(surface_temp)
        dE = E2 - E1
        if dE > 0:                                    # We have a worse new surface
            if np.exp(-dE/3) < np.random.random():    # If much worse, higher chance to disregard
                surface = np.copy(surface_temp)       # Disregard new surface
        if E2 < -198:
            print(iter)
            break
    return surface

def generateTraining(surfSize,setSize):
    surface = randSurface(surfSize)
    set = [0]*setSize
    energy = np.zeros(setSize)
    indices = np.arange(setSize)                    # Used for shuffeling the surfaces
    indices = np.random.permutation(indices)
    for i in range(setSize):
        surface_temp = np.copy(surface)
        shuffle(surface,4)
        E2 = EBondFeatureGrid(surface)
        E1 = EBondFeatureGrid(surface_temp)
        dE = E2 - E1
        set[indices[i]] =np.copy(surface)           # Shuffle surface
        energy[indices[i]] = E2                     # Keep energy in same index number
        if dE > 0:                                  # We have a worse new surface
            if np.exp(-dE/3) < np.random.random():  # If much worse, higher chance to disregard                                       
                surface = np.copy(surface_temp)     # Disregard new surface
        if E2 < -198:                               # We have found the correct surface
            surface = randSurface(surfSize)         # Then reset and continue
    return np.array(set),energy

if __name__ == '__main__':
    import surfaceProject.energycalculations.calcenergy as ce
    import surfaceProject.energycalculations.findStructure as fs
    for i in range(10):
        findOptimum(5)
    surface = findOptimum(5)
    surface2 = fs.findOptimum(5)
    print('The found surface is:',surface)
    print('With energy:',ce.calculateEnergy(surface,5),' and ', EBondFeatureGrid(surface))
    print('With Feature Vector of random atom:',getBondFeatureVectorsSingleGrid(surface)[0])
    print('The correct surface is:',surface2)
    print('With energy:',ce.calculateEnergy(surface2,5),' and ', EBondFeatureGrid(surface2)) 
    print('With FeatureVector of random atom:', getBondFeatureVectorsSingleGrid(surface2)[0])
