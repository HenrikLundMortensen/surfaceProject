import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import surfaceProject.FeatureVector.findStructureWithFeature as fs
import surfaceProject.FeatureVector.featureVector as fv
import surfaceProject.FeatureVector.calcEnergyWithFeature as ce
import surfaceProject.FeatureVector.learningCurve as lc

Ndata_max = 3
X, E = fs.generateTraining(5, Ndata_max)
F = fv.getBondFeatureVectors(X)
(Ng, Na, Nf) = np.shape(F)

motive_list = np.reshape(F, (Ng*Na, Nf))

unique_motives = np.unique(motive_list, axis=0)


N_unique_motives = np.size(unique_motives, axis=0)


def calcLocalEnergy(f_array):
    N = np.size(f_array,0)
    E_array=np.zeros(N)
    for i in range(N):
        E_array[i] = ce.EBondFeature(f_array[i])
    return E_array

if __name__ == '__main__':
    Na = 18  # Number of atoms in unit cell
    Nf = 6  # Number of elements in local feature vector
    Nvalidations = 10  # Nvalidations-fold cross-validation
    Nlearn = 100  # Number of points on each learning curve

    error_array = np.zeros(Nlearn)
    Ndata_max = 10000
    X, E = fs.generateTraining(5, Ndata_max)
    Ndata_array = np.logspace(1, 4, Nlearn).astype(int)
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
            Etrain, Etest = E[i_train], E[i_test]

            Ftrain = fv.getBondFeatureVectors(Xtrain)
                
            # Make list of all unique local features
            Ng_train = np.size(i_train, 0)
            motive_list = np.reshape(Ftrain, (Ng_train*Na, Nf))
            unique_motives = np.unique(motive_list, axis=0)

            # Number of unique features
            N_unique_motives = np.size(unique_motives, axis=0)

            # Make kmeans object with the unique local features as centroids
            kmeans = KMeans(n_clusters=2, random_state=0)
            kmeans.cluster_centers_ = unique_motives

            # Calculate cluster energies
            Ecluster = calcLocalEnergy(unique_motives)
                
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

