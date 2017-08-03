import numpy as np
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt
import calcenergy as ce
N=5
image = ce.randSurface(N)
print(image)
"""                                                                                                                                          
Inputs 3D array f(image,atom,feature_vector)                                                                                                 
                                                                                                                                             
In seperate figures the Ag and O atoms are plotted in the feature space (2D apace of surrounding atomic densies).                            
"""

def plotFeatures(f):
    N_images, N_atoms = size(f,0), size(f,1)
    fori in range(N_images): # Iterate through images                                                                                       
    for j in range(N_atoms): # Iterate througt atoms in a specific image                                                                 
            if f[i,j,2] == 47: # plot Ag                                                                                                     
                plt.figure(1)
plt.plot(f[i,j,0],f[i,j,1])
else f[i,j,2] == 8: # plot O                                                                                                     
    plt.figure(2)
plt.plot(f[i,j,0],f[i,j,1])
