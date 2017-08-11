import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from randomgrid import *
import surfaceProject.FeatureVector.findStructureWithFeature as fs
import surfaceProject.FeatureVector.featureVector as fv
import surfaceProject.FeatureVector.learningCurve as lc
import surfaceProject.FeatureVector.calcEnergyWithFeature as cf

import numpy as np



class plotGridFig:
    """
    Class for plotting grids in various configurations. Generally, a matplotlib figure instance is created. 
    The grid is then plotted by updating existing obejcts in the figure. 

    To plot a single grid with the unit square highlighted, initialize figure with the initializeGridPlot method. 
    To plot a grid, call the plotGrid method. 

    To plot a series of grid next to each other, initialize the figure with the initializeClusterPlot method. 
    To plot a series of grids, call the plotCluster method. 
    """
    
    def __init__(self):

        # Create the matplotlib figure instance
        self.fig = plt.figure()
        self.masterAx = self.fig.gca()
    

    def initializeGridPlot(self,N):
        """
        Initializes the figure to plot a single grid. 
        Input: 
        N: System size (the grid to be plotted must be N x N)
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

        # Create the polygon patch, which highlights the unit cell
        p = patches.Polygon(xy,fc=(0,0,1,0.2),lw=2,ec =(0,0,0,1))

        ax.add_patch(p)

    def initializeClusterPlot(self,N,NumOfRows,NumOfColumns):
        """
        Initialize figure to plot NumOfRows x NumOfColumns grid of size N x N.  

        Input:
        N: System size
        NumOfRows: Number of rows of grids
        NumOfColumns: Number of columns of grids
        """

        
        self.N = N
        self.masterAx.set_axis_off()
        self.NumOfRows = NumOfRows
        self.NumOfColumns = NumOfColumns

        # Specify figure size (Not smart yet!)
        dpi = 20
        featurePlotSizeInPixels = 50
        sepInPixels = int(25/2)

        self.dpi = dpi
        self.featurePlotSizeInPixels = featurePlotSizeInPixels
        self.sepInPixels = sepInPixels

        # Calculate and set width and heigh of image in inches
        widthInInches = ((featurePlotSizeInPixels + sepInPixels)*(NumOfColumns+1) + sepInPixels)/dpi
        heightInInches = ((featurePlotSizeInPixels + sepInPixels)*(NumOfRows+1) + sepInPixels)/dpi
        self.fig.set_size_inches(widthInInches,heightInInches)

        # Create all the axes in where the grids are to be plotted. Save in huge axes matrix
        axMat = np.empty(shape = (NumOfRows,NumOfColumns,2),dtype=object)
        for i in range(NumOfRows):
            for j in range(NumOfColumns):
                widthFactor = 1/(dpi*widthInInches)
                heightFactor = 1/(dpi*heightInInches)
                axesPosition = [ ((featurePlotSizeInPixels + sepInPixels)*(j+1) + sepInPixels )*widthFactor,
                                 ((featurePlotSizeInPixels + sepInPixels)*(i+1) + sepInPixels )*heightFactor,
                                 featurePlotSizeInPixels*widthFactor,
                                 featurePlotSizeInPixels*heightFactor] 

                # The first entry in the [i][j]'th position is the axes itself
                axMat[i][j][0] = plt.axes(axesPosition)

                # No ticks
                axMat[i][j][0].tick_params(bottom='off',left='off')
                axMat[i][j][0].tick_params(labelbottom='off',labelleft='off')

                # Create all the circles. The color and radius are updated when the grids are plotted. 
                p = []
                for m in range(N**2):
                    p.append(patches.Circle(xy = (0,0), radius = 0))
                    axMat[i][j][0].add_artist(p[m])

                # Save the list of patches in the second entry of the [i][j]'th position in axMat
                axMat[i][j][1] = p

                # Add all axes to the figure
                self.fig.add_axes(axMat[i][j][0])

        self.axMat = axMat

    def plotClusters(self,G):
        """
        Updates all the circles created in the InitializeClusterPlot method, according to the grids provided. 

        Input:
        G: Matrix with grids in the entries. The entry G[i][j] is the grid to be plotted in the [i]'th row and 
        [j]'th column. 
        """

        # Run through all entires in G
        for i in range(self.NumOfRows):
            for j in range(self.NumOfColumns):
                axMatEntry = self.axMat[i][j]
                ax = axMatEntry[0]
                g = G[i][j]
                N = g.shape[0]

                # Set facecolor to be purple'ish
                ax.set_facecolor((0.9,0.8,1))

                # Set x and y limits for the plot.
                # Everything is plottede between 0 and 1. Choose limits to be -0.2 and 1.2.
                # The +0.25 is due to the shearing if the grids.
                ax.set_xlim([-0.2+0.25,1.2+0.25])
                ax.set_ylim([-0.2,1.2])

                # Update the [i][j]'th axes
                self.plotGridInAxMatEntry(g,axMatEntry)

                

    def plotGridInAxMatEntry(self,g,axMatEntry):
        """
        Updates the circles created in the InitializeClusterPlot method according to the grid. 

        Input:
        g: Grid to be plotted
        axMatEntry: axMatEntry[0] is the axes where the grid is to be plotted. axMatEntry[1] is a list of
        circles (patches) to be updated accoring to the grid.

        """
        
        # Get system size
        N = g.shape[0]
        g = np.rot90(g,1)
        # Define the shearing matrix
        shearMat = np.array([[1, 0.5],[0,1]])

        # Specify the radius for each atom type
        AgRadius = 1/N*0.95/2
        ORadius = AgRadius/2

        # List of coordinates between 0 and 1
        xcoords = np.linspace(0,1,N)
        ycoords = np.linspace(0,1,N)
        
        m = 0
        for i in range(N):
            for j in range(N):
                
                coord = (xcoords[i],ycoords[j])
                coord = np.dot(shearMat,coord)


                # Update center position and radius according to atom type
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

        neighbourGrid = np.zeros(shape=(3,3))
        neighbourGrid[1][1] = g[i][j]
        neighbourGrid[0][1] = g[np.mod(i-1,N)][j]
        neighbourGrid[1][0] = g[i][np.mod(j-1,N)]
        neighbourGrid[2][1] = g[np.mod(i+1,N)][j]
        neighbourGrid[1][2] = g[i][np.mod(j+1,N)]
        neighbourGrid[2][2] = g[np.mod(i+1,N)][np.mod(j+1,N)]
        neighbourGrid[0][0] = g[np.mod(i-1,N)][np.mod(j-1,N)]

        return neighbourGrid


    def countSubstring(string,sub):
        """
        """
        string = string + string

        count = 0
        i = 0
        while True:
            i = string.find(sub,i) + 1
            if i > 0: 
                count += 1
            else:
                return count

    def neighbourString(nG):
        """
        """
        N = 3
        i = 1
        j = 1
        int_list = [ nG[np.mod(i,N)][np.mod(j-1,N)] ,
                     nG[np.mod(i-1,N)][np.mod(j-1,N)] ,
                     nG[np.mod(i-1,N)][np.mod(j,N)] ,
                     nG[np.mod(i,N)][np.mod(j+1,N)],
                     nG[np.mod(i+1,N)][np.mod(j+1,N)],
                     nG[np.mod(i+1,N)][np.mod(j,N)]]
        return ''.join(str(i) for i in int_list)

    def UniqueNeighbourGridsBeloningToCluster(neighbourGridsBeloningToCluster):
        """
        """
        uniqueNGlistForEachCluster = []
        for nGBTC in neighbourGridsBeloningToCluster:
            uniqueNGlist = []
            for nG in nGBTC:
                nGString = neighbourString(nG)
                count = 0
                for uniqueNG in uniqueNGlist:
                    count += countSubstring(neighbourString(uniqueNG),nGString)

                if count == 0:
                    uniqueNGlist.append(nG)
            
            uniqueNGlistForEachCluster.append(uniqueNGlist)
        return uniqueNGlistForEachCluster
                
            


        

    N = 5
    NTrain = 10000
    NTest = 200
    K = 60

    # Generate training set with Monte Carlo
    X, E = fs.generateTraining(N, NTrain + NTest)

    # Generate random training sest
    XRand = np.array(list(randomgrid(N) for i in range(NTrain + NTest)))
    ERand = np.array(list(cf.EBondFeatureGrid(g) for g in XRand ))
    X = XRand
    E = ERand

    

    # Split into test and training
    Xtrain, Xtest = X[0:NTrain], X[NTrain:NTrain+NTest]
    Etrain, Etest = E[0:NTrain], E[NTrain:NTrain+NTest]

    # Apply clustering
    Ftrain = fv.getBondFeatureVectors(Xtrain)
    [Ftrain_compact, kmeans] = lc.expandedF2compactF(Ftrain, K)

    (Ng, Na, Nf) = np.shape(Ftrain)
    
    # Reshape data for clustering
    F = np.reshape(Ftrain, (Ng*Na, Nf))
    clistMat = kmeans.predict(F)
    clistMat = np.reshape(clistMat, (Ng, Na))

    EClusters = np.dot(np.linalg.pinv(Ftrain_compact),Etrain)

    sortList = np.argsort(EClusters)[::-1]
    EClustersSorted = EClusters[sortList]
    
    neighbourGridsBeloningToCluster = list(range(K))
    for i in range(K):
        neighbourGridsBeloningToCluster[i] = []
        
    for i in range(NTrain):
        indicies = getIndiciesWithAtoms(Xtrain[i])
        clist = clistMat[i]
        indiciesBelongingToCluster = getIndiciesBelongingToCluster(indicies,clist,K)

        for k in range(K):
            if len(indiciesBelongingToCluster[k]) != 0:
                for j in indiciesBelongingToCluster[k]:
                    nG = getNeighbourGrid(Xtrain[i],j[0],j[1])
                    neighbourGridsBeloningToCluster[k].append(nG)

    uniqueNeighbourList = UniqueNeighbourGridsBeloningToCluster(neighbourGridsBeloningToCluster)
    listOfLengths = list(len(a) for a in uniqueNeighbourList)

    NumOfColumns = max(listOfLengths)
    NumOfRows = K
    print(NumOfColumns)
    print(NumOfRows)
    
    G = np.empty(shape=(NumOfRows,NumOfColumns),dtype=object)
    for i in range(NumOfRows):
        for j in range(NumOfColumns):
            if j < listOfLengths[sortList[i]]:
                G[i][j] = uniqueNeighbourList[sortList[i]][j]
            else:
                G[i][j] = np.zeros(shape=(3,3))







                
    figClass = plotGridFig()
    print('I made an instance.')
    
    figClass.initializeClusterPlot(N,NumOfRows,NumOfColumns)
    print('Initialized the figure.')

    figClass.plotClusters(G)
    print('Plotted the local features. Now saving the figure.')

    for i in range(K):
        Estr = "%3.3g" %(EClusters[sortList[i]])
        figClass.axMat[i][0][0].text(-0.2,0.5,Estr,ha='right',va='center',size=50)

    
    figClass.fig.savefig('TestWithRandomGeneratedTrainingset.eps',dpi=int(figClass.dpi*1))
                
            
        
    

    




