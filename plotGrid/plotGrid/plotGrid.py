import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def plotGrid(g,fig):
    """
    Function that plots the grid. 

    First, use initializePlotGridFigure to create a figure instance. The plotGrid function takes this figure as an 
    argument and updates the plots. 
    

    Input:
    g: Grid
    fig: Figure (created with initializePlotGridFigure)

    Output:
    None
    """
    # Rotate grid, such that the view is correct (Maybe not necessary)
    rotg = np.rot90(g,3)

    # Extend the matrix, so we plot 9 unit cells
    tmp = np.concatenate((rotg,rotg),axis=0)
    tmp = np.concatenate((tmp,rotg),axis=0)
    tmp2 = np.concatenate((tmp,tmp),axis=1)
    rotg = np.concatenate((tmp2,tmp),axis=1)

    # Find Ag index and pair them in a list
    indexAg = np.where(rotg==1)
    PairsAg = list( [indexAg[0][i],  indexAg[1][i] ] for i in range(len(indexAg[0])))
    
    # Find O index and pair them in a list
    indexO = np.where(rotg==2)
    PairsO = list( [indexO[0][i],  indexO[1][i] ] for i in range(len(indexO[0])))

    # Define the shearing matrix
    shearMat = np.array([[1, 0.5],[0,1]])

    # Multiply all coordinate pairs with the shearing matrix.
    # The ".T" transposes such that CoordsAg[0] are x coordinates, CoordsAg[1] are y coordinates etc. 
    CoordsAg = np.array(list( np.dot(shearMat,i) for i in PairsAg)).T
    CoordsO = np.array(list( np.dot(shearMat,i) for i in PairsO)).T

    CoordsAg = np.subtract(CoordsAg,5)
    CoordsO = np.subtract(CoordsO,5)
    
    # Update the x and y values of the plots
    ax = fig.gca()

    # ax.get_children()[1] are for the Ag atoms
    ax.get_children()[0].set_xdata(CoordsAg[0])
    ax.get_children()[0].set_ydata(CoordsAg[1])

    # ax.get_children()[2] are for the O atoms
    ax.get_children()[1].set_xdata(CoordsO[0])
    ax.get_children()[1].set_ydata(CoordsO[1])

    # Pause a millisecond, so pyplot updates the graphics
    plt.pause(.001)


def initializePlotGridFigure(N):
    """
    Input:
    N: System size

    Output:
    plt.figure instance
    """
    fig = plt.figure()

    ax = fig.gca()

    # Set background color and x and y limits
    ax.set_facecolor((0.9,0.8,1))
    ax.set_xlim([0,1.5*N+2])
    ax.set_ylim([-2,N])

    
    # Create two plot instances with no data
    ax.plot([],[],'bo',markersize =15)
    ax.plot([],[],'ro',markersize =7)

    return fig



    

    

    


    
    
    

    
    


