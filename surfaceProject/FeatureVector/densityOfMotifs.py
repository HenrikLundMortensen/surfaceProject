import numpy as np
from randomgrid import *
from calcEnergyWithFeature import *
import matplotlib.pyplot as plt
import surfaceProject.FeatureVector.findStructureWithFeature as fs
# import surfaceProject.FeatureVector.featureVector as fv
from surfaceProject.FeatureVector.featureVector import *
from surfaceProject.FeatureVector.learningCurve import *
from surfaceProject.FeatureVector.getClusterNumberMatrix import *
import time





def getClusterNumberMatFromKmeansLabels(labelArray1D, Ng, Na, k):
    # Reshape into seperate grids
    labelArray2D = np.reshape(labelArray1D, (Ng, Na))

    # For each grid; count number of motives belonging to each cluster
    # This is the final feature of a grid.
    clusterNumMat = np.zeros((Ng, k)).astype(int)
    for i in range(Ng):
        clusterNumMat[i, :] = np.bincount(labelArray2D[i, :], minlength=k)
    return clusterNumMat

def clusterNumMatFromKmeans(Ftest, kmeans):
    # Xtest describes each surface with N^2 numbers (atomic identity)
    # Ftest describes each surface with Na x Nf numbers
    # (Nf features for each atom)
    (Ntest, Na, Nf) = np.shape(Ftest)
    k = kmeans.n_clusters
    
    # Reduces the Nf features for each atom into one number
    # describing which cluster it
    # belings to (based on the surrounfing atoms)
    Ftest = np.reshape(Ftest, (Ntest*Na, Nf))
    Ftest = kmeans.predict(Ftest)
    
    clusterNumMat = getClusterNumberMatFromKmeansLabels(Ftest, Ntest, Na, k)
    return clusterNumMat




t1 = time.time()


N = 5
NTotal = 10000

#### Generate large random set
ERand = []
XRand = []

for i in range(NTotal):
    g = randomgrid(N)
    XRand.append(g)
    ERand.append(EBondFeatureGrid(g))

#### Generate large Monte Carlo set
XMC, EMC = fs.generateTraining(N, NTotal)


# Construct all possible local environments

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
################################################## RANDOM
# Generate random grids 

gridsRand = XRand

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
gridMC,E = XMC,EMC

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



# Devide into 15 bins
NBins = 10


bins = np.linspace(-0.01,len(fvUnique),NBins+1)
fvUniqueMCBins = np.zeros(NBins)
fvUniqueRandBins = np.zeros(NBins)
    
for i in range(len(fvUnique)):
    for j in range(NBins):
        if i < bins[j+1]:
            fvUniqueRandBins[j] += fvCountRand[i]            
            fvUniqueMCBins[j] += fvCountMC[i]
            break



################################################## Plot

fig = plt.figure()
ax = fig.gca()
nRand,binsRand,patchesRand = ax.hist(ERand,30,normed=1 ,alpha=0.5)
# ax.bar(bins[0:NBins]+10,fvUniqueRandBins,width = len(fvUnique)/NBins*0.3,alpha = 0.8)
# ax.bar(bins[0:NBins]-10,fvUniqueMCBins,width = len(fvUnique)/NBins*0.3,alpha = 0.8)
ax.plot(bins[0:NBins],fvUniqueRandBins,'b')
ax.plot(bins[0:NBins],fvUniqueMCBins,'r')
ax.set_xlim([-10,len(fvUnique)+10])
ax.set_xlabel('Energy')
ax.set_xticks([])
ax.set_ylabel('Count')
ax.set_title('Motif count')
fig.savefig('MotifCount.png')

t2 = time.time()-t1
print(t2)







