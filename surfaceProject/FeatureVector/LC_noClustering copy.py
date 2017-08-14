import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import surfaceProject.FeatureVector.findStructureWithFeature as fs
import surfaceProject.FeatureVector.featureVector as fv
import surfaceProject.FeatureVector.calcEnergyWithFeature as ce
import surfaceProject.FeatureVector.learningCurve as lc


def calcLocalEnergy3D(f_array):
    N = np.size(f_array, 0)
    E_array = np.zeros(N)
    for i in range(N):
        E_array[i] = ce.EFeature(f_array[i])
    return E_array


def getClusterNumberMatFromKmeansLabels(labelArray1D, Ng, Na, k):
    # Reshape into seperate grids
    labelArray2D = np.reshape(labelArray1D, (Ng, Na))

    # For each grid; count number of motives belonging to each cluster
    # This is the final feature of a grid.
    clusterNumMat = np.zeros((Ng, k)).astype(int)
    for i in range(Ng):
        clusterNumMat[i, :] = np.bincount(labelArray2D[i, :], minlength=k)
    return clusterNumMat


def clusterNumMatFromKmeans3DFeatures(X, kmeans):
    # Xtest describes each surface with N^2 numbers (atomic identity)
    # Ftest describes each surface with Na x Nf numbers
    # (Nf features for each atom)
    F = fv.getFeatureVectors(X)
    (Ng, Na, Nf) = np.shape(F)
    k = np.size(kmeans.cluster_centers_, 0)
    
    # Reduces the Nf features for each atom into one number
    # describing which cluster it
    # belings to (based on the surrounfing atoms)
    F = np.reshape(F, (Ng*Na, Nf))
    F = kmeans.predict(F)
    
    clusterNumMat = getClusterNumberMatFromKmeansLabels(F, Ng, Na, k)
    return clusterNumMat


def calcEnergy3Dfeature(grid_array):
    Ng = np.size(grid_array, 0)
    E_array = np.empty(Ng)
    for i in range(Ng):
        E_array = ce.EFeatureGrid(grid_array[i])
    return E_array

if __name__ == '__main__':
    Na = 18  # Number of atoms in unit cell
    Nf_train = 3  # Number of elements in local feature vector
    Nvalidations = 2  # Nvalidations-fold cross-validation
    Nlearn = 20  # Number of points on each learning curve

    error_array = np.zeros(Nlearn)
    Ndata_max = 1000
    X, E = fs.generateTraining(5, Ndata_max)
    Ndata_array = np.logspace(1, 3, Nlearn).astype(int)
    N_unique_clusters = np.zeros(Nlearn)
    N_unique_motives_array = np.zeros(Nlearn)

    for i in range(Nlearn):
        Ndata = Ndata_array[i]
        Ntest = int(Ndata/Nvalidations)
        Xdata = X[0:Ndata]
        # Do Nvalidations-fold cross-validation
        for j in range(Nvalidations):
            print('%g/10 \t %g/10' %(j, i))

            # Calculate indices for j'th split of data for cross-validation
            [i_train1, i_test, i_train2] = np.split(np.arange(Ndata), [Ntest*j, Ntest*(j+1)])
            i_train = np.r_[i_train1, i_train2]
                
            # Split into test and training
            Xtrain, Xtest = Xdata[i_train], Xdata[i_test]
            Etest = E[i_test]

            # Convert grids to grids of 3D features for training data
            Ftrain = fv.getFeatureVectors(Xtrain)
                
            # Make list of all unique local features
            Ng_train = np.size(i_train, 0)
            motive_list = np.reshape(Ftrain, (Ng_train*Na, Nf_train))
            unique_motives = np.unique(motive_list, axis=0)

            # Number of unique features
            N_unique_motives = np.size(unique_motives, axis=0)

            # Make kmeans object with the unique local features as centroids
            kmeans = KMeans(random_state=0)
            kmeans.cluster_centers_ = unique_motives

            # Calculate cluster energies
            Ecluster = calcLocalEnergy3D(unique_motives)
                
            # Calculate the final feature matrix for the test set
            clusterNumMat_test = lc.clusterNumMatFromKmeans(Xtest, kmeans)

            # Calculate the average squared error between
            # the predicted and actual test set energies.
            Etest_predict = np.dot(clusterNumMat_test, Ecluster)
            error = np.dot(Etest-Etest_predict, Etest-Etest_predict)/Ntest
            error_array[i] += error

            N_unique_clusters[i] += np.size(np.unique(kmeans.cluster_centers_, axis=0), 0)
            N_unique_motives_array[i] += np.size(unique_motives, axis=0)
    error_array /= Nvalidations
    N_unique_clusters /= Nvalidations
    N_unique_motives_array /= Nvalidations
    print("N_unique_clusters\n", N_unique_clusters)
    print("N_unique_motives\n", N_unique_motives_array)
    plt.loglog(Ndata_array, error_array)
    plt.title("Learning Curve")
    plt.xlabel("# training data")
    plt.ylabel("error")
    plt.show()

