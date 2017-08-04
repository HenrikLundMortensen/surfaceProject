import calcenergy as ce
import numpy as np

def countAtomsWithinHexagon(h): # h is array containing the 7 atoms
#    pertub_N_Ag = h.count(1)
#    pertub_N_O = h.count(2)
#    pertub_N_empty = 7-pertub_N_Ag-pertub_N_O
#    return [pertub_N_Ag, pertub_N_O, pertub_N_empty]
    return np.bincount(h)
def nA(grid,i,j,N):
    center = grid[i][j]
    a = grid[np.mod(i,N)][np.mod(j-1,N)] 
    b = grid[np.mod(i-1,N)][np.mod(j-1,N)]
    c = grid[np.mod(i-1,N)][np.mod(j,N)]
    d = grid[np.mod(i,N)][np.mod(j+1,N)]
    e = grid[np.mod(i+1,N)][np.mod(j+1,N)]
    f = grid[np.mod(i+1,N)][np.mod(j,N)]
    return np.array([center,a,b,c,d,e,f])

N=5
surf = ce.randSurface(N)
print("surface:\n", surf)
motive2insert = np.random.randint(3,size=7)
atom_count_pertub = countAtomsWithinHexagon(motive2insert)
print("motive:",motive2insert)
center = np.random.randint(N,size=2)
print("pertubation center:", center)
center_atoms = nA(surf,center[0],center[1],N)
atom_count = countAtomsWithinHexagon(center_atoms)
print(atom_count)

