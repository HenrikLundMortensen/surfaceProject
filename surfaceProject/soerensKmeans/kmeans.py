import numpy as np
import matplotlib.pyplot as plt
import random

def create2dData(N):
    ''' Return an N x 2 array with datapoints between 0 and 1 '''
    return np.random.rand(N,2)
    
def initCentroids(data,N):
    ''' Return N unique centroids given a collection of data points. Input data must be a m x n numpy array, where m is the number of data points and n is the dimension of the data points '''
    
    dataPoints = data.shape[0]
    dim = data.shape[1]
    centroids = np.zeros((N,dim))

    dataPointsUsed = []
    k = 0
    while k < N:
        i = random.randint(0,dataPoints -1)
        if i not in dataPointsUsed:
            centroids[k] = data[i]
            k +=1
            dataPointsUsed.append(i)
    return centroids

def kmeans(data,N):
    '''Create N clusters from the given data. Data must be a m x n numpy array, where m is the number of data points and n is the dimension of the data points'''
    
    numberOfdataPoints = data.shape[0]
    dim = data.shape[1]
    dataPointsCentroid = np.zeros(numberOfdataPoints) # Each element indicate which centroid the given point belongs to
    centroids = initCentroids(data,N)
    test = 0
    
    while True: # Loop until converged
        oldCentroids = np.copy(centroids)
        # Find nearest centroid for each data point
        for i in range(numberOfdataPoints):
            dist = 100000
            iPoint = data[i,:]
            centroidNumber = 0
            for k in range(N):
                cPoint = centroids[k,:]
                newDist = np.linalg.norm(iPoint-cPoint)
                if newDist < dist:
                    dist = newDist
                    centroidNumber = k
            dataPointsCentroid[i] = centroidNumber

        # Calculate new centroid, can do this more efficiently
        for k in range(N):
            center = np.zeros(dim)
            pointsInCenter = 0
            for i in range(numberOfdataPoints):
                if dataPointsCentroid[i] == k:
                    pointsInCenter += 1
                    center += data[i,:]
            center = np.divide(center,pointsInCenter)
            centroids[k,:] = center
        if np.array_equal(oldCentroids,centroids):
            break
        else:
            oldCentroids = centroids
            
    return dataPointsCentroid,centroids
                
if __name__ == '__main__':
    dim = 2
    numberOfdataPoints = 50
    numberOfClusters = 2
    data = create2dData(numberOfdataPoints)

    
    dataPointsCentroid,centroids = kmeans(data,numberOfClusters)
    xCentroid = centroids[:,0]
    yCentroid = centroids[:,1]
    
    # Plot data points
    for i in range(numberOfdataPoints):
        color = int(dataPointsCentroid[i])
        plt.plot(data[i,0],data[i,1],'o',color = 'C' + str(color))

    # Plot centroids 
    plt.plot(xCentroid,yCentroid,'kx')
    plt.show()
