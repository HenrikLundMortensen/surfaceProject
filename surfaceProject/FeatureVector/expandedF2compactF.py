import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from surfaceProject.FeatureVector.findStructureWithFeature import generateTraining

# F[g,a,f], G: grid "g" in dataset, a: atom "a" in grid, f: features for atom
# k clusters
def expandedF2compactF(F, k):
    (Ng,Na,Nf) = np.shape(F)
    
    # Reshape data for clustering
    F = np.reshape(F,(Ng*Na,Nf))

    # Cluster
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(F)

    # Extract centroids and cluster-lables
    centroids = kmeans.cluster_centers_
    F = kmeans.labels_

    # Reshape back into seperate grids
    F = np.reshape(F,(Ng,Na))

    # For each grid; count number of motives belonging to each cluster
    # This is the final feature of a grid.
    F_compact = np.zeros((Ng,k)).astype(int)
    for i in range(Ng):
        F_compact[i,:] = np.bincount(F_compact[i,:])

    return [F_compact, F, centroids]


if __name__ == '__main__':
    Ng = 6
    Na = 4
    k  = 4

    X = np.random.rand(Ng,Na,2)
    
    
    [F_c, F, centroids] = expandedF2compactF(X,k)
    print(F)
    color_array = ["r","b","y","g"]
    for i in range(Ng):
        for j in range(Na):
            plt.plot(X[i,j,0],X[i,j,1],".",color=color_array[F[i,j]])
    plt.plot(centroids[:,0],centroids[:,1],"x",color="k")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("k-means clustering 2D")
    #plt.show()

    Xtrain,Xtest = generateTraining(5,10,0.8)
    print(Xtrain)
    print(Xtest)
