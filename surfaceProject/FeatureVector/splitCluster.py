import surfaceProject.FeatureVector.getClusterNumberMatrix as gcnm
import surfaceProject.Emodel.energyModel as em
import surfaceProject.FeatureVector.findStructureWithFeature as fswf
import surfaceProject.FeatureVector.featureVector as fv
import surfaceProject.plotGrid.plotGrid2 as pg
import numpy as np


class clusterHandler:
    """ Class to cluster data, then split the lowest energy clusters """

    def __init__(self, data, energy, k):
        self.Data = data
        self.Energy = energy
        self.K = k

    def splitClusterEnergy(self):
        # Calculate energy of each cluster
        energyClusters = em.createEnergyModel(self.Clusters, self.Energy)
        
        # Find lowest energy cluster
        minCluster = np.argmin(energyClusters)

        # Find data belonging to this cluster
        (numberOfGrids, numberOfFeatureVectors, numberOfFeatures) = self.Features.shape
        Features = self.Features.reshape(numberOfGrids*numberOfFeatureVectors, numberOfFeatures)
        clusterList = self.Kmeans.predict(Features)
        clusterList = clusterList.reshape(numberOfGrids,numberOfFeatureVectors)

        f = []
        # Find feature vectors belonging t
        for i in range(numberOfGrids):
            for j in range(numberOfFeatureVectors):
                if clusterList[i,j] == minCluster:
                    featureVector = self.Features[i][j]
                    f.append(featureVector)
                    
        f = np.asarray(f)
        print('The minimum cluster is:', minCluster)
        print('The found feature vectors belong to:', self.Kmeans.predict(f))
#        print('FeatureList:',self.Features)
#        print('Feature vectors belonging to minimum:',f)

    def doClustering(self):
        Features = fv.getBondFeatureVectors(self.Data)
        self.Clusters, self.Kmeans = gcnm.getClusterNumberMatrixTraining(Features, self.K)
        self.Clusters = np.asarray(self.Clusters)
        self.Features = Features
    
if __name__ == '__main__':

    # Generate some data
    surfaceSize = 5
    dataSize = 10
    clusters = 7
    Data, Energy = fswf.generateTraining(surfaceSize, dataSize)
    myClusterandler = clusterHandler(Data, Energy, clusters)
    myClusterandler.doClustering()
    myClusterandler.splitClusterEnergy()

