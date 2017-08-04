import numpy as np
import timeit

def randSurface(N): # Creates random surface matrix
    o = 3*N - 9
    ag = N*N - 3*N + 2
    surface = np.zeros(N*N)
    for i in range(0,ag):
        surface[i] = 1
    for i in range(ag,ag+o):
        surface[i] = 2
    surface = np.random.permutation(surface)
    surface = np.reshape(surface,(N,N))
    return surface

def oo(matrix,i,j,size): # Checks for O-O interaction
    E = 0
    if i == size-1:
        i = -1
    if j == size-1:
        j = -1
    if matrix[i,j+1] == 2 and matrix[i-1,j] == 2:
        E += 1
    if matrix[i-1,j] == 2 and matrix[i-1,j-1] == 2:
        E += 1
    if matrix[i-1,j-1] == 2 and matrix[i,j-1] == 2:
        E += 1
    if matrix[i,j-1] == 2 and matrix[i+1,j] == 2:
        E += 1
    if matrix[i+1,j] == 2 and matrix[i+1,j+1] == 2:
        E += 1
    if matrix[i+1,j+1] == 2 and matrix[i,j+1] == 2:
        E += 1
    return E

def agag(matrix,i,j,size): # Checks for Ag-Ag interaction
    E = 0
    if i == size-1:
        i = -1
    if j == size-1:
        j = -1
    if matrix[i,j+1] == 1 and matrix[i-1,j] == 1:
        E -= 1
    if matrix[i-1,j] == 1 and matrix[i-1,j-1] == 1:
        E -= 1
    if matrix[i-1,j-1] == 1 and matrix[i,j-1] == 1:
        E -= 1
    if matrix[i,j-1] == 1 and matrix[i+1,j] == 1:
        E -= 1
    if matrix[i+1,j] == 1 and matrix[i+1,j+1] == 1:
        E -= 1
    if matrix[i+1,j+1] == 1 and matrix[i,j+1] == 1:
        E -= 1
    return E

def ago(matrix,i,j,size): # Checks for Ag-O interaction
    E = 0
    if i == size-1:
        i = -1
    if j == size-1:
        j = -1
    if matrix[i,j+1] == 1 and matrix[i-1,j] != 1 and matrix[i-1,j-1] == 1 and matrix[i,j-1] == 1 and matrix[i+1,j] != 1 and matrix[i+1,j+1] == 1:
        E -= 4
    if matrix[i,j+1] != 1 and matrix[i-1,j] == 1 and matrix[i-1,j-1] == 1 and matrix[i,j-1] != 1 and matrix[i+1,j] == 1 and matrix[i+1,j+1] == 1:
        E -= 4
    if matrix[i,j+1] == 1 and matrix[i-1,j] == 1 and matrix[i-1,j-1] != 1 and matrix[i,j-1] == 1 and matrix[i+1,j] == 1 and matrix[i+1,j+1] != 1:
        E -= 4
    return E


# Calculate the energy
def calculateEnergy(surface,size): # Find size from surface instead
    E = 0
    for i in range(size):        
        for j in range(size):
            if surface[i,j] == 2:
                E += oo(surface,i,j,size)
                E += ago(surface,i,j,size)
            if surface[i,j] == 1:
                E += agag(surface,i,j,size)
    return E

if __name__ == '__main__':
    runs = 100
    size = 5
    t0 = timeit.default_timer()
    for i in range(100):
        surface = randSurface(size)
        E = calculateEnergy(surface,size)
    t1 = timeit.default_timer()
#print('The random surface structure is', surface)
#print('The energy is', E)
    print('The total time for the calculation is',t1-t0, 'seconds')

# Now try with optimized code
    def calculate_single_energy(surface):
        myboard = surface
        myboard_right_neighbor = np.roll(myboard,-1,axis=1)
        myboard_left_neighbor = np.roll(myboard,1,axis=1)
        myboard_upper_right_neighbor = np.roll(myboard,1,axis=0)
        myboard_lower_left_neighbor = np.roll(myboard,-1,axis=0)
        myboard_upper_left_neighbor = np.roll(myboard_left_neighbor,1,axis=0)
        myboard_lower_right_neighbor = np.roll(myboard_right_neighbor,-1,axis=0)

        # Ag triangles
        e1 = -3 * sum(sum((myboard == 1) * (myboard_right_neighbor == 1) * (myboard_upper_right_neighbor == 1)))
        e2 = -3 * sum(sum((myboard == 1) * (myboard_right_neighbor == 1) * (myboard_lower_right_neighbor == 1)))

        # O triangles
        e3 =  3 * sum(sum((myboard == 2) * (myboard_right_neighbor == 2) * (myboard_upper_right_neighbor == 2)))
        e4 =  3 * sum(sum((myboard == 2) * (myboard_right_neighbor == 2) * (myboard_lower_right_neighbor == 2)))

        # O in perfect 4 Ag setup
        e5 =  -4 * sum(sum((myboard == 2) * (myboard_right_neighbor == 1) * (myboard_upper_right_neighbor == 1) * (myboard_upper_left_neighbor != 1) * (myboard_left_neighbor == 1) * (myboard_lower_left_neighbor == 1) * (myboard_lower_right_neighbor != 1) ))
        e6 =  -4 * sum(sum((myboard == 2) * (myboard_right_neighbor != 1) * (myboard_upper_right_neighbor == 1) * (myboard_upper_left_neighbor == 1) * (myboard_left_neighbor != 1) * (myboard_lower_left_neighbor == 1) * (myboard_lower_right_neighbor == 1) ))
        e7 =  -4 * sum(sum((myboard == 2) * (myboard_right_neighbor == 1) * (myboard_upper_right_neighbor != 1) * (myboard_upper_left_neighbor == 1) * (myboard_left_neighbor == 1) * (myboard_lower_left_neighbor != 1) * (myboard_lower_right_neighbor == 1) ))
        return e1+e2+e3+e4+e5+e6+e7

    t0 = timeit.default_timer()
    for i in range(100):
        surface = randSurface(size)
        E = calculate_single_energy(surface)
    t1 = timeit.default_timer()
    print('Now trying with more advanced algorithm')
#print('The energy is',E)
    print('The total time for the calculation is',t1-t0,'seconds')
