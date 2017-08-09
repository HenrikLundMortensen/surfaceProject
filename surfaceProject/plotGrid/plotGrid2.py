import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from randomgrid import *
import numpy as np





class plotGridFig:

    
    def __init__(self):
        self.fig = plt.figure()
        self.masterAx = self.fig.gca()
        self.fig.set_size_inches(10,10)
        

    def show(self):
        self.fig.show()
    
    

    def initializeGridPlot(self,N):
        """
        Input:
        N: System size
        """

        ax = self.masterAx

        # Set background color and x and y limits
        ax.set_facecolor((0.9,0.8,1))
        ax.set_xlim([0-(N-1)*0.5,(N-1)*2])
        ax.set_ylim([0-(N-1)*0.75,(N-1)*1.75])
        plt.xticks([])
        plt.yticks([])

        # Create two plot instances with no data
        # Decrease markersize as N increases
        self.AgPlot = ax.plot([],[],'bo',markersize =5*15*1/N)[0]
        self.OPlot = ax.plot([],[],'ro',markersize =5*7*1/N)[0]

        # Define the shearing matrix
        shearMat = np.array([[1, 0.5],[0,1]])

        # Draw rectangle
        xshift=0
        yshift=0
        x = [0+xshift,0+xshift,(N-1)+xshift,(N-1)+xshift]
        y = [0+yshift,N-1+yshift,N-1+yshift,0+yshift]
        xy = np.array(list(zip(x,y)))

        # Apply shearing to coordinates
        for i in range(4):
            xy[i] = np.dot(shearMat,xy[i])

        # xy = np.array([[1,5],[2,6],[3,7],[4,8]])
        p = patches.Polygon(xy,fc=(0,0,1,0.2),lw=2,ec =(0,0,0,1))

        ax.add_patch(p)

    def initializeClusterPlot(self,N,NumOfClusters,NumOfVisuals):
        """
        Input:
        N: System size
        """

        self.N = N
        self.masterAx.set_axis_off()
        self.NumOfClusters = NumOfClusters
        self.NumOfVisuals = NumOfVisuals

        
        dpi = 50
        featurePlotSizeInPixels = 100
        sepInPixels = 25

        self.dpi = dpi
        self.featurePlotSizeInPixels = featurePlotSizeInPixels
        self.sepInPixels = sepInPixels

        widthInInches = ((featurePlotSizeInPixels + sepInPixels)*NumOfVisuals + sepInPixels)/dpi
        heightInInches = ((featurePlotSizeInPixels + sepInPixels)*NumOfClusters + sepInPixels)/dpi
        self.fig.set_size_inches(widthInInches,heightInInches)

        axMat = np.empty(shape = (NumOfClusters,NumOfVisuals,2),dtype=object)
        
        for i in range(NumOfClusters):
            for j in range(NumOfVisuals):
                widthFactor = 1/(dpi*widthInInches)
                heightFactor = 1/(dpi*heightInInches)
                axesPosition = [ ((featurePlotSizeInPixels + sepInPixels)*j + sepInPixels )*widthFactor,
                                 ((featurePlotSizeInPixels + sepInPixels)*i + sepInPixels )*heightFactor,
                                 featurePlotSizeInPixels*widthFactor,
                                 featurePlotSizeInPixels*heightFactor] 

                axMat[i][j][0] = plt.axes(axesPosition)
                axMat[i][j][0].tick_params(bottom='off',left='off')
                axMat[i][j][0].tick_params(labelbottom='off',labelleft='off')

                p = []
                for m in range(N**2):
                    p.append(patches.Circle(xy = (0,0), radius = 0))
                    axMat[i][j][0].add_artist(p[m])
                    
                axMat[i][j][1] = p
                self.fig.add_axes(axMat[i][j][0])

        self.axMat = axMat

    def plotClusters(self,G):


        for i in range(self.NumOfClusters):
            for j in range(self.NumOfVisuals):
                axMatEntry = self.axMat[i][j]
                ax = axMatEntry[0]
                g = G[i][j]
                N = g.shape[0]
                
                ax.set_facecolor((0.9,0.8,1))
                ax.set_xlim([-0.1+0.25,1.1+0.25])
                ax.set_ylim([-0.1,1.1])

                self.plotGridInAxMatEntry(g,axMatEntry)

                

    def plotGridInAxMatEntry(self,g,axMatEntry):
        # Get system size
        N = g.shape[0]

        # Define the shearing matrix
        shearMat = np.array([[1, 0.5],[0,1]])

        AgRadius = 1/N*0.95/2
        ORadius = AgRadius/2

        xcoords = np.linspace(0,1,N)
        ycoords = np.linspace(0,1,N)
        m = 0
        for i in range(N):
            for j in range(N):
                
                coord = (xcoords[i],ycoords[j])
                coord = np.dot(shearMat,coord)
                
                if g[i][j] == 0:
                    axMatEntry[1][m].center = coord
                    axMatEntry[1][m].radius = 0

                if g[i][j] == 1:
                    axMatEntry[1][m].center = coord
                    axMatEntry[1][m].radius = AgRadius
                    axMatEntry[1][m].set_color('blue')
                    

                if g[i][j] == 2:
                    axMatEntry[1][m].center = coord
                    axMatEntry[1][m].radius = ORadius                    
                    axMatEntry[1][m].set_color('red')
                    
                m += 1
                

        
    
    def plotGrid(self,g):        
        """
        Method that plots the grid. 

        First, use initializePlotGridFigure to create a figure instance. The plotGrid function takes this figure as an 
        argument and updates the plots. 
        
        Input:
        g: Grid

        Output:
        None
        """
        # Get system size
        N = g.shape[0]
    
        # Rotate grid, such that the view is correct (Maybe not necessary)
        rotg = np.rot90(g,3)


        # Extend the matrix, so we plot 9 unit cells
        tmp = np.concatenate((rotg,rotg),axis=0)
        tmp = np.concatenate((tmp,tmp),axis=0)
        tmp2 = np.concatenate((tmp,tmp),axis=1)
        rotg = np.concatenate((tmp2,tmp2),axis=1)

        # Find Ag index and pair them in a list
        indexAg = np.where(rotg==1)
        PairsAg = list( [indexAg[0][i],  indexAg[1][i] ] for i in range(len(indexAg[0])))

        # Find O index and pair them in a list
        indexO = np.where(rotg==2)
        PairsO = list( [indexO[0][i],  indexO[1][i] ] for i in range(len(indexO[0])))

        # Define the shearing matrix
        shearMat = np.array([[1, 0.5],[0,1]])

        PairsAg = np.subtract(PairsAg,(N*2))
        PairsO = np.subtract(PairsO,(N*2))

        # Multiply all coordinate pairs with the shearing matrix.
        # The ".T" transposes such that CoordsAg[0] are x coordinates, CoordsAg[1] are y coordinates etc. 
        CoordsAg = np.array(list( np.dot(shearMat,i) for i in PairsAg)).T
        CoordsO = np.array(list( np.dot(shearMat,i) for i in PairsO)).T

        # CoordsAg = np.subtract(CoordsAg,(N-1))
        # CoordsO = np.subtract(CoordsO,(N-1))

        # Update the x and y values of the plots
        self.AgPlot.set_xdata(CoordsAg[0])
        self.AgPlot.set_ydata(CoordsAg[1])
        self.OPlot.set_xdata(CoordsO[0])
        self.OPlot.set_ydata(CoordsO[1])

        # Pause a millisecond, so pyplot updates the graphics
        plt.pause(.001)
        


if __name__ == '__main__':


    def getIndiciesWithAtoms(g):
        
        N = g.shape[0]

        indicies = []
        for i in range(N):
            for j in range(N):
                if g[i][j] != 0:
                    indicies.append((i,j))

        return np.array(indicies)

    def getIndiciesBelongingToCluster(indicies,clist,K):

        indiciesBelongingToCluster = []
        for k in range(K):
            indiciesBelongingToCluster.append(indicies[np.where(clist == k)])

        return np.array(indiciesBelongingToCluster)
        
    

    def getNeighbourGrid(g,i,j):
        N = g.shape[0]
        print(g)
        neighbourGrid = np.zeros(shape=(3,3))
        neighbourGrid[1][1] = g[i][j]
        neighbourGrid[0][1] = g[np.mod(i-1,N)][j]
        neighbourGrid[1][0] = g[i][np.mod(j-1,N)]
        neighbourGrid[2][1] = g[np.mod(i+1,N)][j]
        neighbourGrid[1][2] = g[i][np.mod(j+1,N)]
        neighbourGrid[2][2] = g[np.mod(i+1,N)][np.mod(j+1,N)]
        neighbourGrid[0][0] = g[np.mod(i-1,N)][np.mod(j-1,N)]

        return neighbourGrid
        
        



    
    
    N = 5
    figClass = plotGridFig()
    figClass.initializeClusterPlot(N,8,10)


    G = np.empty(shape=(8,10),dtype=object)
    for i in range(8):
        for j in range(10):
            G[i][j] = randomgrid(N)

    figClass.plotClusters(G)

    figClass.fig.savefig('mytest.png',dpi=figClass.dpi)
    print((getNeighbourGrid(G[0][0],2,2)))
    
    # for i in range(10):
    #     g = randomgrid(N)
    #     figClass.plotGrid(g)
    #     plt.pause(1)

    # ind = getIndiciesWithAtoms(G[0][0])
    # clist = np.array([0,0,2,0,1,0,0,0,2,0,0,0,0,0,0,0,0,0])
    # K = 10

    # print(len(getIndiciesBelongingToCluster(ind,clist,K)[4]))



    

    # print(getIndiciesWithAtoms(G[0][0]))
    




                
            
        
    

    




