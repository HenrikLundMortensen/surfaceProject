import numpy as np
import matplotlib.pyplot as plt

# Determine initial k cluster centers (cc) at random
def initialize_centers(data,N,k):
    cluster_centers_index = np.arange(N)
    cluster_centers_index = np.random.permutation(cluster_centers_index)
    cluster_centers_index= cluster_centers_index[0:k]
    cluster_centers = X[cluster_centers_index,:]
    return cluster_centers

#cc = initialize_centers(X,N,k)

# Assign each datapoint to nearest cluster
def assign_centers(data, centers):
    N = np.size(data,0)
    k = np.size(centers,0)
    dist_closest=np.zeros(N)
    assigned_cluster = np.zeros(N).astype(int)
    for i in range(N):
        dist_closest[i] = np.dot(X[i,:]-centers[0,:],X[i,:]-centers[0,:])
        assigned_cluster[i] = 0
        for j in range(1,k):
            dist = np.dot(X[i,:]-centers[j,:],X[i,:]-centers[j,:])
            if dist < dist_closest[i]:
                dist_closest[i] = dist
                assigned_cluster[i] =j
    return assigned_cluster

#assigned_cluster = assign_centers(X,cc)

# Calculate the centers as the new cluster mean
def evolve_centers(data,assigned_cluster,centers):
    N = np.size(data,0)
    k = np.size(centers,0)
    new_centers = np.copy(centers)
    for i in range(N):
        new_centers[assigned_cluster[i]] += data[i,:]
    points_in_clusters = np.bincount(assigned_cluster)
    new_centers = np.divide(new_centers,points_in_clusters.reshape((6,1))+1)
    return new_centers

# k-means algoritme
def kmeans(X,k):
    N = np.size(X,0)
    C = initialize_centers(X,N,k)
    while True:
        # Assign each point to nearest centroid (outputs index of assigned centroid)
        C_assigned = assign_centers(X,C).astype(int)

        # Calculate new centroids as cluster-mean (including centroid)
        C_new = evolve_centers(X,C_assigned,C)

        # Terminate when the centroids does not change
        if np.array_equal(C_new,C):
            break
        C = np.copy(C_new)
    return [C , C_assigned]


if __name__ == '__main__':
    N = 20 # Number of data points
    k = 6 # Number of clusters

    # Create random feature vector (X)
    X = np.random.rand(N,2)

    # Apply k-means
    [cc, cc_assigned] = kmeans(X,k)

    # plot data (assigned to clusters) + centroids
    color_array = ["r","b","y","g","c","m"]
    for i in range(N):
        plt.plot(X[i,0],X[i,1],"o",color=color_array[cc_assigned[i]])
    plt.plot(cc[:,0],cc[:,1],"x",color="k")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("k-means clustering 2D")
    plt.show()
