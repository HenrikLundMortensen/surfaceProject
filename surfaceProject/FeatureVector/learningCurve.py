import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import surfaceProject.FeatureVector.findStructureWithFeature as fs
import surfaceProject.FeatureVector.featureVector as fv


def getClusterNumberMatFromKmeansLabels(labelArray1D, Ng, Na, k):
    # Reshape into seperate grids
    labelArray2D = np.reshape(labelArray1D, (Ng, Na))

    # For each grid; count number of motives belonging to each cluster
    # This is the final feature of a grid.
    clusterNumMat = np.zeros((Ng, k)).astype(int)
    for i in range(Ng):
        clusterNumMat[i, :] = np.bincount(labelArray2D[i, :], minlength=k)
    return clusterNumMat


# F[g,a,f], G: grid "g" in dataset, a: atom "a" in grid, f: features for atom
# k clusters
def expandedF2compactF(F, k):
    (Ng, Na, Nf) = np.shape(F)
    
    # Reshape data for clustering
    F = np.reshape(F, (Ng*Na, Nf))

    # Cluster
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(F)

    # Extract cluster-lables
    F = kmeans.labels_

    # Reshape back into seperate grids
    F = np.reshape(F, (Ng, Na))

    # For each grid; count number of motives belonging to each cluster
    # This is the final feature of a grid.
    ClusterNumMat = np.zeros((Ng, k)).astype(int)
    for i in range(Ng):
        ClusterNumMat[i, :] = np.bincount(F[i, :], minlength=k)

    return [ClusterNumMat, kmeans]


def getClusterEnergies(Xtrain, Etrain):
    return np.dot(np.linalg.pinv(Xtrain), Etrain)


def clusterNumMatFromKmeans(Xtest, kmeans):
    # Xtest describes each surface with N^2 numbers (atomic identity)
    # Ftest describes each surface with Na x Nf numbers
    # (Nf features for each atom)
    Ftest = fv.getBondFeatureVectors(Xtest)
    (Ntest, Na, Nf) = np.shape(Ftest)
    
    # Reduces the Nf features for each atom into one number
    # describing which cluster it
    # belings to (based on the surrounfing atoms)
    Ftest = np.reshape(Ftest, (Ntest*Na, Nf))
    Ftest = kmeans.predict(Ftest)
    
    clusterNumMat = getClusterNumberMatFromKmeansLabels(Ftest, Ntest, Na, k)
    return clusterNumMat


if __name__ == '__main__':
    """
    Ng = 6
    Na = 4
    k  = 4

    X = np.random.rand(Ng,Na,2)
    
    
    [F_c, kmeans] = expandedF2compactF(X,k)
    centroids = kmeans.cluster_centers_
    F = kmeans.labels_
    F = np.reshape(F,(Ng,Na))
    #print(F_c)
    color_array = ["r","b","y","g"]
    for i in range(Ng):
        for j in range(Na):
            plt.plot(X[i,j,0],X[i,j,1],".",color=color_array[F[i,j]])
    plt.plot(centroids[:,0],centroids[:,1],"x",color="k")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("k-means clustering 2D")
    plt.show()
    """
    
    k = 60
    Na = 18
    Nf = 6
    Ntrain_max = 2000
    Ntest = 100
    error_array = np.zeros(10)
    for j in range(10):
        X, E = fs.generateTraining(5, Ntrain_max+Ntest)
        Ndata_array = np.logspace(1, 3, 10)
        for i in range(10):
            print('%g/10 \t %g/10' %(j,i))
            Ntrain = Ndata_array[i].astype(int)
    
            # Split into test and training
            Xtrain, Xtest = X[0:Ntrain], X[Ntrain:Ntrain+Ntest]
            Etrain, Etest = E[0:Ntrain], E[Ntrain:Ntrain+Ntest]

            # Apply clustering
            Ftrain = fv.getBondFeatureVectors(Xtrain)
            [Ftrain_compact, kmeans] = expandedF2compactF(Ftrain, k)

            # Calculate cluster energies
            Ecluster = getClusterEnergies(Ftrain_compact, Etrain)
    
            clusterNumMat_test = clusterNumMatFromKmeans(Xtest, kmeans)
    
            Etest_predict = np.dot(clusterNumMat_test, Ecluster)
            error = np.dot(Etest-Etest_predict, Etest-Etest_predict)/Ntest
            error_array[i] += error
    error_array /= 10
    plt.loglog(Ndata_array, error_array)
    plt.title("Learning Curve")
    plt.xlabel("# training data")
    plt.ylabel("error")
    plt.show()
