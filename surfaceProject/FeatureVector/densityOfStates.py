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
K = 60

NTotal = 100000
NTrain = 1000
NTest = NTotal-NTrain

#### Generate large random set
ERand = []
XRand = []

for i in range(NTotal):
    g = randomgrid(N)
    XRand.append(g)
    ERand.append(EBondFeatureGrid(g))

#### Generate large Monte Carlo set
XMC, EMC = fs.generateTraining(N, NTotal)

#### Split into training and test set
XRandTrain , ERandTrain = XRand[0:NTrain],ERand[0:NTrain]
XMCTrain , EMCTrain = XMC[0:NTrain],EMC[0:NTrain]

XRandTest , ERandTest = XRand[NTrain:NTrain+NTest],ERand[NTrain:NTrain+NTest]
XMCTest , EMCTest = XMC[NTrain:NTrain+NTest],EMC[NTrain:NTrain+NTest]

#### Merge MC and Rand test sets into one large set. Divide this set into bins according to the energy
XTest = []
ETest = []

for i in range(NTrain):
    XTest.append(XMCTest[i])
    ETest.append(EMCTest[i])
for i in range(NTrain):
    XTest.append(XRandTest[i])    
    ETest.append(ERandTest[i])

# Sort the set according to the their energy

sortList = np.argsort(ETest)
ETestSorted = np.array(ETest)[sortList]
XTestSorted = np.array(XTest)[sortList]

# Divide the set into 10 bins
NTestBins = 10
bins = np.linspace(min(ETest)-0.01,max(ETest)+0.01,NTestBins+1)
indexList = []
XTestBins = []
ETestBins = []
ETestBinsListForHist = []
for i in range(NTestBins):
    XTestBins.append([])
    ETestBins.append([])
    
for i in range(len(ETestSorted)):
    for j in range(NTestBins):
        if ETestSorted[i] < bins[j+1]:
            indexList.append(j)
            if len(XTestBins[j]) < 2000000:
                XTestBins[j].append(XTestSorted[i])
                ETestBins[j].append(ETestSorted[i])
                ETestBinsListForHist.append(ETestSorted[i])
            break
    
#### Get machine learning models from training sets
# Starting with Rand
# Apply clustering
FRandTrain = getBondFeatureVectors(XRandTrain)
[FRandTrain_compact, kmeansRand] = expandedF2compactF(FRandTrain, K)

# Calculate cluster energies
ERandcluster = getClusterEnergies(FRandTrain_compact, ERandTrain)

# And now MC
# Apply clustering
FMCTrain = getBondFeatureVectors(XMCTrain)
[FMCTrain_compact, kmeansMC] = expandedF2compactF(FMCTrain, K)

# Calculate cluster energies
EMCcluster = getClusterEnergies(FMCTrain_compact, EMCTrain)

#### Get error from each of the test set in the bins
errorRandList = []
errorMCList = []
indexList = []
for i in range(len(XTestBins)):

    XTests = XTestBins[i]
    ETests = ETestBins[i]
    
    if len(XTests) > 0:

        FTest = fv.getBondFeatureVectors(np.array(XTests))
        clusterNumMatRand = clusterNumMatFromKmeans(FTest,kmeansRand)
        clusterNumMatMC = clusterNumMatFromKmeans(FTest, kmeansMC)

        # Predict energies

        EPredictRand = np.dot(clusterNumMatRand, ERandcluster)
        EPredictMC = np.dot(clusterNumMatMC, EMCcluster)

        errorRand = np.dot(ETests-EPredictRand, ETests-EPredictRand)/len(ETests)
        errorMC = np.dot(ETests-EPredictMC, ETests-EPredictMC)/len(ETests)
        errorRandList.append(errorRand)
        errorMCList.append(errorMC)
        indexList.append(i)


        
fig = plt.figure()
ax = fig.gca()


nRand,binsRand,patchesRand = ax.hist(ERand,30,normed=1 ,alpha=0.5)
nMonteC,binsMonteC,patchesMonteC = ax.hist(EMC,30,normed=1, alpha=0.5)
nTest,binsTest,patchesTest = ax.hist(ETestBinsListForHist,30,normed=1 ,alpha=0.5)

axright = ax.twinx()
axright.semilogy(np.array(bins)[indexList],errorRandList)
axright.semilogy(np.array(bins)[indexList],errorMCList)
axright.set_ylim([min([min(errorRandList),min(errorMCList)])/4 ,max([max(errorRandList),max(errorMCList)])*8])
axright.set_ylabel('Error')

ax.legend(['Random','Monte Carlo','Test'])
ax.set_xlabel('Energy')
ax.set_ylabel('Count')
ax.set_title('Density of states')

ax.text(0.06,0.90,"Sample size = %g" %(NTotal),size=10,transform=ax.transAxes)
ax.text(0.06,0.85,"Training set size = %g" %(NTrain),size=10,transform=ax.transAxes)
fig.savefig('densityOfStatesWithError.png')
t2 = time.time()-t1
print(t2)







############################## Motif count plot is made below ##############################

# # Construct all possible local environments

# strlist = []
# for a in range(3):
#     for b in range(3):
#         for c in range(3):
#             for d in range(3):
#                 for e in range(3):
#                     for f in range(3):
#                         strlist.append(''.join(str(m) for m in [a,b,c,d,e,f]))



# # Calculate local grid and feature vectors for each local environment

# gridList = []
# fvList = []

# for string in strlist:
#     for s in [1,2]:

#         # Get grid
#         grid = np.zeros(shape=(3,3))
#         grid[1][1] = s
#         grid[0][1] = int(string[0])
#         grid[0][0] = int(string[1])
#         grid[1][0] = int(string[2])        
#         grid[1][2] = int(string[3])
#         grid[2][2] = int(string[4])
#         grid[2][1] = int(string[5])

#         gridList.append(grid)

#         # Get feature vector
#         # Get neighbouring atoms                                                                                                          
#         neighbours = list(int(string[i]) for i in range(6))
        
#         # Calculate the densities                                                                                                         
#         [Od,Agd] = calcDensities(neighbours)

#         if s==1:
#             a = 47

#         if s==2:
#             a = 8

#         [ooLength,agagLength,agoLength] = calcBondLength(neighbours)

#         fvList.append([Od,Agd,a,ooLength,agagLength,agoLength])



# # Get unique feature vectors, their energies and store the index.

# fvUnique = []
# EUnique = []
# indexList = []
# index = 0
# for fv in fvList:
#     count = 0
#     for ufv in fvUnique:
#         if ufv == fv:
#             count +=1


#     if count == 0:
#         fvUnique.append(fv)
#         indexList.append(index)
#         EUnique.append(EBondFeature(fv))

#     index +=1
# uniqueGrids = np.array(gridList)[indexList]


# sortList = np.argsort(EUnique)
# EUniqueSorted = np.array(EUnique)[sortList]
# fvUniqueSorted = np.array(fvUnique)[sortList]
# gridsUniqueSorted = uniqueGrids[sortList]




# # For the set of grids, calculate how many times a unique feature appears

# Ntest = 1000
# ################################################## RANDOM
# # Generate random grids 

# gridsRand = []
# for i in range(Ntest):
#     gridsRand.append(randomgrid(N))

# # Find all feature vectors
# Flist = []
# for grid in gridsRand:
#     Flist.append(getBondFeatureVectorsSingleGrid(grid))

# Flist = np.array(Flist)
# Flist = np.reshape(Flist, (Flist.shape[0]*Flist.shape[1],Flist.shape[2]) )


# fvCountRand = np.zeros(len(fvUnique))

# m = 0
# for uf in fvUniqueSorted:
#     for f in Flist:
#         if np.array_equal(f,uf):
#             fvCountRand[m] +=1
#     m += 1
            
# ################################################## Monte Carlo
# # Generate from MC
# gridMC,E = fs.generateTraining(N, Ntest)

# # Find all feature vectors
# Flist = []
# for grid in gridMC:
#     Flist.append(getBondFeatureVectorsSingleGrid(grid))

# Flist = np.array(Flist)
# Flist = np.reshape(Flist, (Flist.shape[0]*Flist.shape[1],Flist.shape[2]) )


# fvCountMC = np.zeros(len(fvUnique))

# m = 0
# for uf in fvUniqueSorted:
#     for f in Flist:
#         if np.array_equal(f,uf):
#             fvCountMC[m] +=1
#     m += 1




# ################################################## Plot


# fig = plt.figure()
# ax = fig.gca()
# ax.plot(fvCountRand,'b*')
# ax.plot(fvCountMC,'r*')
# ax.set_xlabel('Energy')
# ax.set_ylabel('Count')
# ax.set_title('Motif count')
# fig.savefig('MotifCount.png')
