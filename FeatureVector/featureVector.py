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

def calcBondLength(nA): # calc O-O, Ag-Ag and Ag-O bond length of surrounding atoms
    ooLength = 0
    agagLength = 0
    agoLength = 0
    for i in range(len(nA)-1):
        if nA[i] != 0:
            for j in range(i+1,len(nA)):
                length = 0
                if nA[j] != 0:
                    if j - i > 3:
                        length = 6-(j-i)
                    else:
                        length = j-i
                    if length == 2:
                        length = np.sqrt(3)
                    if length == 3:
                        length = 2
                    if nA[i] != nA[j]:
                        agoLength += np.exp(-length)
                    elif nA[i] == 1:
                        agagLength += np.exp(-length)
                    else: ooLength  += np.exp(-length)
    return [ooLength, agagLength,agoLength]

def getBondFeatureVectorsSingleGrid(grid):
    # Get size                                                                                                                                    

    N = np.size(grid,0)


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
                [ooLength,agagLength,agoLength] = calcBondLength(neighbours)

                f.append([Od,Agd,a,ooLength,agagLength,agoLength])

    return np.array(f)

def getFeatureVectorsSingleGrid(grid):

    # Get size

    N = np.size(grid,0)


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

def getBondFeatureVectors(G):

    # Define list of list of feature vectors - initially empty                                                                                    
    f = []

    # Get feature vectors for each grid                                                                                                           
    for g in G:
        f.append(getBondFeatureVectorsSingleGrid(g))


    return np.array(f)

if __name__ == '__main__':
    testArray = [2,2,2,2,2,2]
    testLength = calcBondLength(testArray)
    print(testLength)
    
            


    
    
    


