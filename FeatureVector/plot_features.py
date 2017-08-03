import numpy as np
import matplotlib.pyplot as plt
import calcenergy as ce

"""                                                                                                                                          
Inputs 3D array f(image,atom,feature_vector)                                                                                                 
                                                                                                                                             
In seperate figures the Ag and O atoms are plotted in the feature space (2D apace of surrounding atomic densies).                            
"""
        
def plot_features(f):
    N_images, N_atoms = np.size(f,0), np.size(f,1)
    for i in range(N_images): # Iterate through images 
        for j in range(N_atoms): # Iterate througt atoms in a specific image
            if f[i,j,2] == 47: # plot Ag
                plt.figure(1)
                plt.plot(f[i,j,0],f[i,j,1],"o",color="blue")
            elif f[i,j,2] == 8: # plot O
                plt.figure(2)
                plt.plot(f[i,j,0],f[i,j,1],"o",color="red")
    plt.figure(1)
    plt.title("Feature space of Ag atoms")
    plt.xlabel("density of O atoms")
    plt.ylabel("density of Ag atoms")
    plt.figure(2)
    plt.title("Feature space of Ag atoms")
    plt.xlabel("density of O atoms")
    plt.ylabel("density of Ag atoms")
    plt.show()
