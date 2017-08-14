import surfaceProject.FeatureVector.clusterHandler as ch
import surfaceProject.FeatureVector.findStructureWithFeature as fswf
import surfaceProject.FeatureVector.featureVector as fv
import surfaceProject.Emodel.energyModel as em
import surfaceProject.FeatureVector.getClusterNumberMatrix as gcnm
import numpy as np
import matplotlib.pyplot as plt


class LearningCurve:
    """
    Description goes here
    """

    def __init__(self, dataSize, clusters, surfaceSize):
        self.K = clusters
        self.DataSize = dataSize
        self.SurfaceSize = surfaceSize
        self.Data, self.Energy = fswf.generateTraining(surfaceSize, dataSize)
        
    def generateDataPoint(self, trainingSize):

        # Split data into training and testing
        dataTrain, energyTrain = self.Data[:trainingSize], self.Energy[:trainingSize]
        dataTest, energyTest = self.Data[trainingSize:], self.Energy[trainingSize:]

        # Do the clustering
        myClusterhandler = ch.ClusterHandler(dataTrain, energyTrain, self.K)
        myClusterhandler.doClustering()

        # Calculate energy of each cluster
        energyClusters = em.createEnergyModel(myClusterhandler.Clusters, myClusterhandler.Energy)

        # Find feature vectors in test set and convert to clusters in the test set
        featuresInTest = fv.getBondFeatureVectors(dataTest)
        clustersInTest = gcnm.getClusterNumberMatrix(featuresInTest, myClusterhandler.Kmeans)
        
        # Predict energy of each surface in the test set and calculate error
        energyPredict = em.getEnergyFromModel(clustersInTest, energyClusters)
        error = np.dot(energyTest - energyPredict, energyTest - energyPredict) / (self.DataSize - trainingSize)
        return error

    def generateCurve(self, startSet, endSet, increment):
        self.ErrorList = []
        self.ErrorListL = []
        self.TrainingList = []
        for i in range(startSet, endSet, increment):
            errorPoint = self.generateDataPoint(i)
            errorPointL = self.generateDataPointWithLargeClusterSplitting(i)
            self.ErrorListL.append(errorPointL)
            self.ErrorList.append(errorPoint)
            self.TrainingList.append(i)
        self.ErrorList = np.array(self.ErrorList)
        self.ErrorListL = np.array(self.ErrorListL)
        self.TrainingList = np.array(self.TrainingList)

    def plotCurve(self):
        plt.plot(self.TrainingList, self.ErrorList, 'ro', label='Normal clustering')
        plt.plot(self.TrainingList, self.ErrorListL, 'go', label='Clustering with splitting')
        plt.xlabel('Training size')
        plt.ylabel('Error')
        plt.legend()
        plt.show()

    def generateDataPointWithLowEnergySplitting(self, trainingSize):
        # Split data into training and testing
        dataTrain, energyTrain = self.Data[:trainingSize], self.Energy[:trainingSize]
        dataTest, energyTest = self.Data[trainingSize:], self.Energy[trainingSize:]

        # Do the clustering by creating two clusters and then splitting until K clusters are achieved
        myClusterhandler = ch.ClusterHandler(dataTrain, energyTrain, 2)
        myClusterhandler.doClustering()
        while myClusterhandler.K < self.K:
            myClusterhandler.splitClusterLowEnergy()

        # Calculate energy of each cluster
        energyClusters = em.createEnergyModel(myClusterhandler.Clusters, myClusterhandler.Energy)

        # Find feature vectors in test set and convert to clusters in the test set
        featuresInTest = fv.getBondFeatureVectors(dataTest)
        clustersInTest = gcnm.getClusterNumberMatrix(featuresInTest, myClusterhandler.Kmeans)
        
        # Predict energy of each surface in the test set and calculate error
        energyPredict = em.getEnergyFromModel(clustersInTest, energyClusters)
        error = np.dot(energyTest - energyPredict, energyTest - energyPredict) / (self.DataSize - trainingSize)
        return error

    def generateDataPointWithLargeClusterSplitting(self, trainingSize):
        # Split data into training and testing
        dataTrain, energyTrain = self.Data[:trainingSize], self.Energy[:trainingSize]
        dataTest, energyTest = self.Data[trainingSize:], self.Energy[trainingSize:]

        # Do the clustering by creating half the clusters and then splitting until K clusters are achieved
        myClusterhandler = ch.ClusterHandler(dataTrain, energyTrain, int(self.K / 2))
        myClusterhandler.doClustering()
        while myClusterhandler.K < self.K:
            myClusterhandler.splitClusterLargest()

        # Calculate energy of each cluster
        energyClusters = em.createEnergyModel(myClusterhandler.Clusters, myClusterhandler.Energy)

        # Find feature vectors in test set and convert to clusters in the test set
        featuresInTest = fv.getBondFeatureVectors(dataTest)
        clustersInTest = gcnm.getClusterNumberMatrix(featuresInTest, myClusterhandler.Kmeans)
        
        # Predict energy of each surface in the test set and calculate error
        energyPredict = em.getEnergyFromModel(clustersInTest, energyClusters)
        error = np.dot(energyTest - energyPredict, energyTest - energyPredict) / (self.DataSize - trainingSize)
        return error

        
if __name__ == '__main__':
    clusters = 150
    datasize = 10000
    surfacesize = 5
    start, end, increment = 100, datasize, 10000
    myLearningCurve = LearningCurve(datasize, clusters, surfacesize)

    # Plot learning curves
    myLearningCurve.generateCurve(start, end, increment)
    myLearningCurve.plotCurve()
    
    
