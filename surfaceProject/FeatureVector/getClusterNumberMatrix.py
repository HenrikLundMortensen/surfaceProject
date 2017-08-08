import numpy as np
from scipy.cluster.vq import kmeans2
from sklearn.cluster import KMeans
from surfaceProject.FeatureVector.featureVector import *
from surfaceProject.FeatureVector.findStructureWithFeature import *
from surfaceProject.henriksKmeans.kmeans import * 


def randomgrid(n):

    no = 3*n-9
    nag = n**2 - 3*n + 2
    nh = 7

    a = np.array(nh*[0] + no* [2] + nag *[1])
    np.random.shuffle(a)
    b = a.reshape(n,n)
    return b



def getClusterNumberMatrix(F,K):
    """
    Takes a set of feature vectors, clusters them into K clusters and return a vector with how
    many of a cluster type that appears for each grid

    Input:
    F: Numpy array with feature vectors for several grids. On the form F[grid][atom][feature]
    K: Number of clusters

    Output:
    Numpy array with number of each cluster type for each grid
    """
    


    # Number of grids, feature vectors and features
    (G,Ng,Nf)= F.shape

    # Reshape F into a single large array containing feature vectors
    F = F.reshape(G*Ng,Nf)

    # Find K clusters using K-means
    # [fclist,clusterList] = kmeans2(F,K)

    kmeans = KMeans(n_clusters = K).fit(F)

    clusterList = kmeans.labels_

    # Reshape into G x Ng matrix
    clusterList = clusterList.reshape(G,Ng)


    # For each grid, count how many of a cluster type it contains
    CNmatrix = []
    for c in clusterList:
        CNmatrix.append(np.bincount(c))

    return np.array(CNmatrix)


if __name__ == '__main__':

    N = 10
    # Get training set
    [G,T] = generateTraining(N,10,0.8)
    F = getBondFeatureVectors(G)
    print(getClusterNumberMatrix(F,10))


    


    
    
    
    
