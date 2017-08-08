import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.vq import kmeans2
import time
    

def featureVecDist(f1,f2):
    """
    Calculate distance between feature vectors f1 and f2
    """

    dist = 0

    for i in range(len(f1)):
        dist += (f1[i]-f2[i])**2
        
    return dist

        

def randomInitialCenter(flist,K):
    """
    """

    # Number of features
    Nf = len(flist[0])
    
    # Determine boundaries
    bmax = np.empty(Nf)
    bmin = np.empty(Nf)
    for i in range(Nf):
        bmax[i] = np.max(np.transpose(flist)[i])
        bmin[i] = np.min(np.transpose(flist)[i])

    # Generate K random feature vectors
    rflist = []
    for k in range(K):
        rf = []
        for i in range(Nf):
            rf.append( (np.random.rand()*(bmax[i]-bmin[i]))  + bmin[i]  )
    
        rflist.append(rf)
            
    return np.array(rflist)


def randInitFlist(flist,K):
    """
    """

    initFlist = []
    rlist = []

    
    while len(rlist) < K:
        r = np.random.randint(len(flist))
        if r not in rlist:
            rlist.append(r)
            initFlist.append(flist[r])

    return initFlist



def henrikskmeans(flist,K):
    """
    Finds K clusters in list of feature vectors.

    Input:
    flist: numpy array with featurevectors, such that flist = np.array([f1,f2,f3,...])
    K: Number of clusters

    Output: [clist,fcs]
    
    clist: Contains an number representing which cluster the n'th feature vector belongs to.
    If feature vector 1 and 3 belongs to first cluster and feature vector 2 belongs to the second, clist is [0,1,0]


    fcs: Cluster centers, [clustercenter_1,clustercenter_2,...]
    """
    
    # Get some starting point for feature centers
    fcs = randInitFlist(flist,K)
    
    clist = [np.array(list(np.argmin(list(featureVecDist(f,fc) for fc in fcs)) for f in flist)),
             'Some placeholder string']

    # While the clusters change between iterations
    while not np.array_equal(clist[0],clist[1]):
        
        # Find new centers
        for k in range(K):
            cflist = flist[np.where(clist[0]==k)]
            fcs[k] = 1/len(cflist) * sum(cflist)

        # Calculate new clusters
        clist = [np.array(list(np.argmin(list(featureVecDist(f,fc) for fc in fcs)) for f in flist)),
                 clist[0]]

    return [clist[0],fcs]

