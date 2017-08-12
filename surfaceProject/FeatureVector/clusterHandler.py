import surfaceProject.FeatureVector.getClusterNumberMatrix as gcnm
import surfaceProject.Emodel.energyModel as em
import surfaceProject.FeatureVector.findStructureWithFeature as fswf
import surfaceProject.FeatureVector.featureVector as fv
import numpy as np


class clusterHandler:
    """ Class to handle clustering. Can split the cluster in several ways.

    #### Input ####
    data: 3D numpy array of the form [Grid,i,j] where (i,j) describes the element at this position
    in the grid. Either 0,1 or 2

    Energy: list of energies for the above mentioned grids.

    k: the number of clusters one wishes to create

    #### Attributes ####
    self.Data, self.Energy and self.K are the above mentioned quantaties

    self.Clusters: a Grid x clusters list displaying the number of atoms in each cluster for each grid
    
    self.Features: a [Grid][Atom][Features] array displaying feature vectors for each atom in each grid

    self.Kmeans: the Kmeans returned from scipy. Note that intertia is not updated (maybe implement later!)

    #### Methods ####
    doClustering: do Kmeans clustering

    splitClusterLowEnergy: split the cluster with the lowest energy into two clusters

    
    """

    def __init__(self, data, energy, k):
        self.Data = data
        self.Energy = energy
        self.K = k

    def splitClusterLowEnergy(self):
        # Calculate energy of each cluster
        energyClusters = em.createEnergyModel(self.Clusters, self.Energy)
        
        # Find lowest energy cluster
        minCluster = np.argmin(energyClusters)

        # Find data belonging to this cluster
        (numberOfGrids, numberOfFeatureVectors, numberOfFeatures) = self.Features.shape
        Features = self.Features.reshape(numberOfGrids * numberOfFeatureVectors, numberOfFeatures)
        clusterList = self.Kmeans.predict(Features)
        clusterList = clusterList.reshape(numberOfGrids, numberOfFeatureVectors)

        # Find the feature vectors
        f = []
        for i in range(numberOfGrids):
            for j in range(numberOfFeatureVectors):
                if clusterList[i, j] == minCluster:
                    featureVector = self.Features[i][j]
                    f.append(featureVector)
        f = [f]  # Put it into a "grid" to make it work with getClusterNumberMatrixTraining
        f = np.asarray(f)

        # Now cluster this into two new clusters
        newClusters, newKmeans = gcnm.getClusterNumberMatrixTraining(f, 2)
        newCentroids = newKmeans.cluster_centers_

        # Now add them to the existing clusters
        self.Kmeans.cluster_centers_ = np.delete(self.Kmeans.cluster_centers_, minCluster, 0)
        self.Kmeans.cluster_centers_ = np.append(self.Kmeans.cluster_centers_, newCentroids, 0)
        self.Kmeans.labels_ = self.Kmeans.predict(Features)
        
        # For each grid count how many cluster type it contains
        clusterList = self.Kmeans.labels_.reshape(numberOfGrids, numberOfFeatureVectors)
        CNmatrix = []
        for c in clusterList:
            CNmatrix.append(np.bincount(c))
        self.Clusters = np.asarray(CNmatrix)

    def doClustering(self):
        Features = fv.getBondFeatureVectors(self.Data)
        self.Clusters, self.Kmeans = gcnm.getClusterNumberMatrixTraining(Features, self.K)
        self.Clusters = np.asarray(self.Clusters)
        self.Features = Features


        
if __name__ == '__main__':

    # Generate some data
    surfaceSize = 5
    dataSize = 10
    clusters = 3
    Data, Energy = fswf.generateTraining(surfaceSize, dataSize)
    myClusterandler = clusterHandler(Data, Energy, clusters)
    myClusterandler.doClustering()

    # Calculate energy of the clusters
    energyClusters = em.createEnergyModel(myClusterandler.Clusters, myClusterandler.Energy)
    print('The energy of the clusters are:', energyClusters)
    print('The number of atoms in each cluster are:', myClusterandler.Clusters)

    # Now do splitting
    myClusterandler.splitClusterLowEnergy()
    energyClusters = em.createEnergyModel(myClusterandler.Clusters, myClusterandler.Energy)
    print('The energy of the clusters after splitting are:', energyClusters)
    print('The number of atoms in each cluster after splitting are:', myClusterandler.Clusters)

    # Try splitting again
    myClusterandler.splitClusterLowEnergy()
    energyClusters = em.createEnergyModel(myClusterandler.Clusters, myClusterandler.Energy)
    print('The energy of the clusters after splitting twice are:', energyClusters)
    print('The number of atoms in each cluster after splitting twice  are:', myClusterandler.Clusters)
