import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.vq import kmeans2
import time


    

def featureVecDist(f1,f2):
    """
    Calculate distance between feature vectors f1 and f2
    """

    dist = 0

    for i in range(len(f1)):
        dist += (f1[i]-f2[i])**2
        
    return dist

        

def randomInitialCenter(flist,K):
    """
    """

    # Number of features
    Nf = len(flist[0])
    
    # Determine boundaries
    bmax = np.empty(Nf)
    bmin = np.empty(Nf)
    for i in range(Nf):
        bmax[i] = np.max(np.transpose(flist)[i])
        bmin[i] = np.min(np.transpose(flist)[i])

    # Generate K random feature vectors
    rflist = []
    for k in range(K):
        rf = []
        for i in range(Nf):
            rf.append( (np.random.rand()*(bmax[i]-bmin[i]))  + bmin[i]  )
    
        rflist.append(rf)
            
    return np.array(rflist)


def randInitFlist(flist,K):
    """
    """

    initFlist = []
    rlist = []

    
    while len(rlist) < K:
        r = np.random.randint(len(flist))
        if r not in rlist:
            rlist.append(r)
            initFlist.append(flist[r])

    return initFlist


def getNearestCenter(f,fcs):
    """
    For feature f calculate the nearest center in fcs list
    """

    
    dist = np.array([])
    for fc in fcs:
        dist = np.append(dist,featureVecDist(f,fc))

        
    return np.argmin(dist)
    

def getIntermediateClusters(flist,fcs):
    """
    """

    clist = []
    for f in flist:
        clist.append(getNearestCenter(f,fcs))

    return np.array(clist)


def getNewCenter(cflist):
    """
    """

    return 1/len(cflist) * sum(cflist)


    
def Mykmeans(flist,K):
    """
    """

    # Get some starting point for feature centers
    fcs = randInitFlist(flist,K)

    clist = [np.array(list(np.argmin(list(featureVecDist(f,fc) for fc in fcs)) for f in flist)),
             'Some placeholder value']
    
    i = 0
    # clistref = np.zeros(len(flist))
    # clist = np.random.rand(len(flist))
    while not np.array_equal(clist[0],clist[1]):

        # Calculate current clusters
        

        # Stop if it has not changed since last iteration
        # if :
        #     break

        # If not, save a copy of the current cluster list
        # clistref = clist

        # Calculate new feature centers
        for k in range(K):
            fcs[k] = getNewCenter(flist[np.where(clist[0] == k)])

        clist = [getIntermediateClusters(flist,fcs),clist[0]]
        
        i += 1
        
    return [clist,fcs]






def kmeanscompact(flist,K):
    """
    """

    fig = plt.figure()
    ax = fig.gca()

    
    # Get some starting point for feature centers
    fcs = randInitFlist(flist,K)

    Nf = len(flist[0])

    
    clist = [np.array(list(np.argmin(list(featureVecDist(f,fc) for fc in fcs)) for f in flist)),
             'Some placeholder value']


    colorlist = ['bo','ro','yo']
    pltlist = [0]*K
    for k in range(K):
        pltlist[k], =ax.plot(np.transpose(flist[np.where(clist[0] == k)])[0],np.transpose(flist[np.where(clist[0] == k)])[1],colorlist[k],markersize=5)

    fcsplt, = ax.plot(np.transpose(fcs)[0],np.transpose(fcs)[1],'k*',markersize=20)
    
    plt.pause(1)
    
    # clist = [np.array(list(np.argmin(list(featureVecDist(f,fc) for fc in fcs)) for f in flist)),
    #          'Some placeholder value']


    # While the clusters change between iterations
    i = 0
    while not np.array_equal(clist[0],clist[1]):
        
        # Find new centers
        for k in range(K):
            cflist = flist[np.where(clist[0]==k)]
            fcs[k] = 1/len(cflist) * sum(cflist)

        
        # Calculate new clusters
        t1 = time.time()
        clist = [np.array(list(np.argmin(list(featureVecDist(f,fc) for fc in fcs)) for f in flist)),
                 clist[0]]
        t2 = time.time()
        print('Calculate new clusters:%g ' %(t2-t1))


        # plot

        for k in range(K):
            pltlist[k].set_xdata(np.transpose(flist[np.where(clist[0] == k)])[0])
            pltlist[k].set_ydata(np.transpose(flist[np.where(clist[0] == k)])[1])

        fcsplt.set_xdata(np.transpose(fcs)[0])
        fcsplt.set_ydata(np.transpose(fcs)[1])

        plt.pause(1)
            
        



        
        i += 1
        
    return [clist[0],fcs]


# TESTING


        



# Generate two groups of feature vectors with two features for testing
flist = []

# First group 

for i in range(500):
    flist.append([np.random.rand()*(17-5)  + 5  ,np.random.rand()*(17-5)  + 5 ])
        

# Second group
for i in range(500):
    flist.append([np.random.rand()*(20-15)  + 15  ,np.random.rand()*(20-12)  + 12 ])


flist = np.array(flist)

K = 2


# fig = plt.figure()
# ax = fig.gca()
# ax.plot(np.transpose(flist)[0],np.transpose(flist)[1],'bo',markersize=5)


a = time.time()
for i in range(1):
    kmeans2(flist,3)
t1 = time.time()-a




a = time.time()
for i in range(1):
    kmeanscompact(flist,3)
t2 = time.time()-a


# [clist,fcs] = kmeanscompact(flist,K)
# [fcs2,clist2] = kmeans2(flist,K)
[fcs,clist] = kmeans2(flist,K)


# print(t1*10)
# print(t2*10)
# print(it1)
# print(it2)





# for i in range(10):
    
#     [clist,fcs] = kmeans(flist,K)


#     plt.cla()
# colorlist = ['bo','ro','yo']
# for k in range(K):
#     ax.plot(np.transpose(flist[np.where(clist == k)])[0],np.transpose(flist[np.where(clist == k)])[1],colorlist[k],markersize=5)
#     fcsplt, = ax.plot(np.transpose(fcs)[0],np.transpose(fcs)[1],'k*',markersize=20)

        
# #     plt.pause(0.4)

# print('Done!')
# plt.show()




