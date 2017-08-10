import numpy as np
from randomgrid import *
from calcEnergyWithFeature import *
import matplotlib.pyplot as plt
import surfaceProject.FeatureVector.findStructureWithFeature as fs
from surfaceProject.FeatureVector.featureVector import *

N = 5

Ntest = 20000
ERand = []
for i in range(Ntest):
    g = randomgrid(N)
    ERand.append(EBondFeatureGrid(g))


Ntest = 20000
X, EMonteC = fs.generateTraining(N, Ntest)

fig = plt.figure()
ax = fig.gca()
nRand,binsRand,patchesRand = ax.hist(ERand,30,normed=1,alpha=0.5)
nMonteC,binsMonteC,patchesMonteC = ax.hist(EMonteC,30,normed=1,alpha=0.5)
ax.set_xlabel('Energy')
ax.set_ylabel('Normalized count')
ax.set_title('Density of states')
ax.text(0.06,0.8,"Sample size = %g" %(Ntest),size=10,transform=ax.transAxes)
fig.savefig('densityOfStates.png')




# Construc all possible local environments

strlist = []
for a in range(3):
    for b in range(3):
        for c in range(3):
            for d in range(3):
                for e in range(3):
                    for f in range(3):
                        strlist.append(''.join(str(m) for m in [a,b,c,d,e,f]))



# Calculate local grid and feature vectors for each local environment

gridList = []
fvList = []
for string in strlist:
    for s in [1,2]:

        # Get grid
        grid = np.zeros(shape=(3,3))
        grid[1][1] = s
        grid[0][1] = int(string[0])
        grid[0][0] = int(string[1])
        grid[1][0] = int(string[2])        
        grid[1][2] = int(string[3])
        grid[2][2] = int(string[4])
        grid[2][1] = int(string[5])

        gridList.append(grid)

        # Get feature vector
        # Get neighbouring atoms                                                                                                          
        neighbours = list(int(string[i]) for i in range(6))
        
        # Calculate the densities                                                                                                         
        [Od,Agd] = calcDensities(neighbours)

        if s==1:
            a = 47

        if s==2:
            a = 8

        [ooLength,agagLength,agoLength] = calcBondLength(neighbours)

        fvList.append([Od,Agd,a,ooLength,agagLength,agoLength])



# Get unique feature vectors, their energies and store the index.

fvUnique = []
EUnique = []
indexList = []
index = 0
for fv in fvList:
    count = 0
    for ufv in fvUnique:
        if ufv == fv:
            count +=1


    if count == 0:
        fvUnique.append(fv)
        indexList.append(index)
        EUnique.append(EBondFeature(fv))

    index +=1
uniqueGrids = np.array(gridList)[indexList]


sortList = np.argsort(EUnique)
EUniqueSorted = np.array(EUnique)[sortList]
fvUniqueSorted = np.array(fvUnique)[sortList]
gridsUniqueSorted = uniqueGrids[sortList]



# For the set of grids, calculate how many times a unique feature appears

Ntest = 1000
################################################## RANDOM
# Generate random grids 

gridsRand = []
for i in range(Ntest):
    gridsRand.append(randomgrid(N))

# Find all feature vectors
Flist = []
for grid in gridsRand:
    Flist.append(getBondFeatureVectorsSingleGrid(grid))

Flist = np.array(Flist)
Flist = np.reshape(Flist, (Flist.shape[0]*Flist.shape[1],Flist.shape[2]) )


fvCountRand = np.zeros(len(fvUnique))

m = 0
for uf in fvUniqueSorted:
    for f in Flist:
        if np.array_equal(f,uf):
            fvCountRand[m] +=1
    m += 1
            
################################################## Monte Carlo
# Generate from MC
gridMC,E = fs.generateTraining(N, Ntest)

# Find all feature vectors
Flist = []
for grid in gridMC:
    Flist.append(getBondFeatureVectorsSingleGrid(grid))

Flist = np.array(Flist)
Flist = np.reshape(Flist, (Flist.shape[0]*Flist.shape[1],Flist.shape[2]) )


fvCountMC = np.zeros(len(fvUnique))

m = 0
for uf in fvUniqueSorted:
    for f in Flist:
        if np.array_equal(f,uf):
            fvCountMC[m] +=1
    m += 1




################################################## Plot


fig = plt.figure()
ax = fig.gca()
ax.plot(fvCountRand,'b*')
ax.plot(fvCountMC,'r*')
ax.set_xlabel('Energy')
ax.set_ylabel('Count')
ax.set_title('Motif count')
fig.savefig('MotifCount.png')
