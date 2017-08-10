import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import surfaceProject.FeatureVector.findStructureWithFeature as fs
import surfaceProject.FeatureVector.featureVector as fv

Ndata_max = 3
X, E = fs.generateTraining(5, Ndata_max)
F = fv.getBondFeatureVectors(X)
(Ng, Na, Nf) = np.shape(F)

motive_list = np.reshape(F, (Ng*Na, Nf))
unique_motives = np.unique(motive_list, axis=0)
N_unique_motives = np.size(unique_motives, axis=0)


Y = np.c_[np.arange(9), np.arange(9)]
print(Y)
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.cluster_centers_ = Y

print(kmeans.cluster_centers_)
print(kmeans.predict([0.1,0.2]))
