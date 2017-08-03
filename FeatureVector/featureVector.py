import numpy as np

"""
Use getFeatureVectors to get feature vectors for an array of grids

Input: Array of grids (must be numpy arrays)
Output: numpy Array f with the following format:

f[grid][atom][feature]

So f[9][1][0] is the first feature of the second atom in the 10'th grid




The features are
[0]: O density
[1]: Ag density
[2]: Atomic number
"""

def nA(grid,i,j,N):
    a = grid[np.mod(i,N)][np.mod(j-1,N)] 
    b = grid[np.mod(i-1,N)][np.mod(j-1,N)]
    c = grid[np.mod(i-1,N)][np.mod(j,N)]
    d = grid[np.mod(i,N)][np.mod(j+1,N)]
    e = grid[np.mod(i+1,N)][np.mod(j+1,N)]
    f = grid[np.mod(i+1,N)][np.mod(j,N)]
    return [a,b,c,d,e,f]

def calcDensities(nA):

    # O density
    Od = nA.count(2)*np.exp(-1)

    # Ag density
    Agd = nA.count(1)*np.exp(-1)

    return [Od,Agd]


def getFeatureVectorsSingleGrid(grid):

    # Get size
    N = grid.shape[0]

    # Define list of feature vectors - initially empty
    f = []

    # For all atoms calculate and append a feature vector
    for i in range(N):
        for j in range(N):
            s = grid[i][j]
            if s!=0:
                # Get neighbouring atoms
                neighbours = nA(grid,i,j,N)

                # Calculate the densities
                [Od,Agd] = calcDensities(neighbours)

                if s==1:
                    a = 47

                if s==2:
                    a = 8
                
                f.append([Od,Agd,a])

    return np.array(f)


def getFeatureVectors(G):

    # Define list of list of feature vectors - initially empty
    f = []

    # Get feature vectors for each grid
    for g in G:
        f.append(getFeatureVectorsSingleGrid(g))


    return np.array(f)


    
            

    
    
    
    


