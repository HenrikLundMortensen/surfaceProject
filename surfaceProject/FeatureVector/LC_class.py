import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import surfaceProject.FeatureVector.findStructureWithFeature as fs
import surfaceProject.FeatureVector.featureVector as fv


'''
Input:
Data in the form of a list of "images" and their corresponding energies "E".
Nval-fold cross-validation
Nk clusters. If Nk = 0, each unique motiv is used as a cluster center
''' 


class ML_clusterModel:

    def __init__(self, Images, energy, Nval, Nk, Nf):
        self.images = Images
        self.energy = energy
        self.Nval = Nval
        self.Nk = Nk
        self.Nf = Nf
        self.Ndata = np.size(self.images, 0)

    def MLwithCrossVal(self):
        self.grid2features()
        error_array = np.empty(self.Nval)

        # Cross-validation
        for i in range(self.Nval):
            # split into training and test set
            self.Ntest = int(np.floor(self.Ndata / self.Nval))
            self.Ntrain = self.Ndata - self.Ntest
            [i_train1, i_test, i_train2] = np.split(np.arange(self.Ndata),
                                                    [self.Ntest * i, self.Ntest * (i + 1)])
            i_train = np.r_[i_train1, i_train2]
            Ftrain, Ftest = self.Feature[i_train], self.Feature[i_test]
            Etrain, Etest = self.energy[i_train], self.energy[i_test]

            Ftrain = self.prepareForClustering(Ftrain, self.Ntrain)

            # Train model
            if Nk == 0:
                self.noCluster(Ftrain, Etrain)
            else:
                self.cluster(Ftrain, Etrain)
                
            # Test model
            error_array[i] = self.predict(Ftest, Etest)
        self.validation_error = np.sum(error_array) / self.Nval

    def grid2features(self):
        # calculate local features
        if self.Nf == 3:
            self.Feature = fv.getFeatureVectors(self.images)
        else:
            self.Feature = fv.getBondFeatureVectors(self.images)
        self.Na = np.size(self.Feature, 1)
        
    def prepareForClustering(self, F, Ndata):
        # The local features are reshaped into an Ni*Na (Na is number of atoms in unit/cell)
        # long list for clustering algorithm, which clusters local features.

        # reshape to array of local features for clustering
        F = np.reshape(F, (Ndata * self.Na, self.Nf))
        return F
        
    def cluster(self, F, E):
        self.kmeans = KMeans(n_clusters=self.Nk)
        self.kmeans.fit(F)

        # Extract labeled atoms from clustering
        F = np.reshape(self.kmeans.labels_, (self.Ntrain, self.Na))

        # Calculate Nmat (matrix containing Number of atoms belonging to each cluster for
        # each image.)
        Nmat = np.zeros((self.Ntrain, self.Nk)).astype(int)
        for i in range(self.Ntrain):
            Nmat[i, :] = np.bincount(F[i, :], minlength=self.Nk)

        # Calculate cluster energies by solving E = Nmat*Ecluster
        # (in least squares sense using pseudo-inverse)
            self.Ecluster = np.dot(np.linalg.pinv(Nmat), E)
            
    def noCluster(self, F, E):
        # Determine unique motives
        F_unique = np.unique(F, axis=0)
        self.Nk = np.size(F_unique, 0)
        
        # Use Unique motives as cluster centers
        self.kmeans = KMeans(random_state=0)
        self.kmeans.cluster_centers_ = F_unique
        
        # Extract labeled atoms from clustering
        F = np.reshape(self.kmeans.predict(F), (self.Ntrain, self.Na))

        # Calculate Nmat (matrix containing Number of atoms belonging to each cluster for
        # each image.)
        Nmat = np.zeros((self.Ntrain, self.Nk)).astype(int)
        for i in range(self.Ntrain):
            Nmat[i, :] = np.bincount(F[i, :], minlength=self.Nk)

        # Calculate cluster energies by solving E = Nmat*Ecluster
        # (in least squares sense using pseudo-inverse)
        self.Ecluster = np.dot(np.linalg.pinv(Nmat), E)

    def predict(self, F, E):
        # Calculate Nmat (matrix containing Number of atoms belonging to each cluster for
        # each image.)
        F = self.prepareForClustering(F, self.Ntest)
        F = np.reshape(self.kmeans.predict(F), (self.Ntest, self.Na))
        Nmat = np.zeros((self.Ntest, self.Nk)).astype(int)
        for i in range(self.Ntest):
            Nmat[i, :] = np.bincount(F[i, :], minlength=self.Nk)
        Epredict = np.dot(Nmat, self.Ecluster)
        error = np.dot(E - Epredict, E - Epredict) / self.Ntest
        return error


# Demonstration of class
if __name__ == '__main__':
    Nk = 60
    Nf = 6
    Nval = 10
    Npoints = 10  # number of points on learning curve
    Ndata_max = 1000
    Itot, Etot = fs.generateTraining(5, Ndata_max)
    Ndata_array = np.logspace(1, 3, Npoints).astype(int)
    print(Ndata_array)
    error_array = np.zeros(Npoints)
    for i in range(Npoints):
        print('%i/%i' %(i, Npoints))
        I = Itot[0:Ndata_array[i]]
        E = Etot[0:Ndata_array[i]]
        model = ML_clusterModel(I, E, Nval, Nk, Nf)
        model.MLwithCrossVal()
        error_array[i] = model.validation_error

    plt.loglog(Ndata_array, error_array)
    plt.title("Learning Curve")
    plt.xlabel("# training data")
    plt.ylabel("error")
    plt.show()
