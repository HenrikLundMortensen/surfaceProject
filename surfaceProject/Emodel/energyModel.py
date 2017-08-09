import numpy as np
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
import scipy.linalg as linalg
from matplotlib import pyplot as plt
import surfaceProject.FeatureVector.findStructureWithFeature as fs
import surfaceProject.FeatureVector.featureVector as fv
from surfaceProject.FeatureVector.getClusterNumberMatrix import *


def createEnergyModel(CNmatrix,Elist):
    """
    Input:
    CNmatrix: Cluster number matrix
    Elist: Energy list

    Output:
    EC: Energy associated with each cluster
    """

    EC = np.dot(linalg.pinv(CNmatrix),Elist)
    return EC


def getEnergyFromModel(CN,EC):
    """
    Input:
    CN: Number of each clusters
    EC: Energy associated with each cluster

    Output:
    Emodel: Energy predicted by the model
    """

    Emodel = np.dot(CN,EC)
    return Emodel
    


######################################## TESTING ########################################


if __name__ == '__main__':


    # Grid size
    N = 5
    
    # Number of clusters
    K = 10

    errorlist = []
    i = 0;
    
    for NTrain in [100,200,300,400,500,600,700,800]:
        print(i)
        error = []
        
        # Size of set
        NSet = 1000

        # Size of training set
        # NTrain = 80
        NTest = 200

        # Generate traning set
        I, E = fs.generateTraining(N,NSet)

        ITrain = I[0:NTrain]
        ETrain = E[0:NTrain]
        ITest = I[NSet-NTest:NSet]
        ETest = E[NSet-NTest:NSet]

        FTrain = fv.getBondFeatureVectors(ITrain)
        FTest = fv.getBondFeatureVectors(ITest)
        

        clusterNumMatTrain,kmeans_result = getClusterNumberMatrixTraining(FTrain,K)
        clusterNumMatTest = getClusterNumberMatrix(FTest,kmeans_result,K)    
    
        EC = createEnergyModel(clusterNumMatTrain,ETrain)

        EModel = np.dot(clusterNumMatTest,EC)
        error = np.dot(ETest-EModel,ETest-EModel)/NTrain
        errorlist.append(error)
        i += 1
        
    
