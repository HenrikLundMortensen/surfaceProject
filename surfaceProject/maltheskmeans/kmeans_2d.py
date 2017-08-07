import numpy as np

N = 20 # Number of data points
k = 6 # Number of clusters

# Create random feature vector (X)
X = np.random.rand(N,2)

# Determine initial k cluster centers (cc) at random
cc_index = np.random.randint(N,size=k)
cc = X[cc_index][:]

print(X)
print(cc_index)
print(cc)

dist_closest=np.zeros(N)
center_closest = np.zeros(N)
for i in range(N):
    dist_closest[i] = np.abs(np.sum(np.dot(X[i],cc[0])))
    center_closest[0] = 0
    for j in range(1,k):
        dist = np.abs(np.sum(np.dot(X[i],cc[j])))
        if dist < dist_closest[i]:
            dist_closest[i] = dist
            center_closest[i] =j

print(center_closest)
print(dist_closest)



