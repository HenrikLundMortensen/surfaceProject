import numpy as np
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
import scipy.linalg as linalg
from matplotlib import pyplot as plt


def createEnergyModel(CNmatrix,Elist):
    """
    Input:
    CNmatrix: Cluster number matrix
    Elist: Energy list

    Output:
    EC: Energy associated with each cluster
    """

    EC = np.dot(linalg.pinv(cM),EV)
    return EC


def getEnergyFromModel(CN,EC):
    """
    Input:
    CN: Number of each clusters
    EC: Energy associated with each cluster

    Output:
    Emodel: Energy predicted by the model
    """

    Emodel = np.dot(Nclist,EW)
    return Emodel
    


######################################## TESTING ########################################


if __name__ == '__main__':


    # Number of clusters
    K = 10
    errorlist = []
    for NT in list(range(K,1000)):
        error = []
        # Energy vector
        EV = np.random.rand(NT)
        cM = np.random.rand(NT,K)
        EW = createEnergyModel(cM,EV)
        Emodel = getEnergyFromModel(cM[0],EW)
        for i in range(NT):
            error.append((EV[i]-Emodel)**2)

        errorlist.append(sum(error)/NT)
        
    plt.plot(list(range(K,1000)),errorlist)
    plt.show()
    
