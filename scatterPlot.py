import sys
#sys.path.append('/Users/Soren/Work/Introduction/surfaceProject') # 
import FeatureVector.featureVector as fv
import FeatureVector.plotFeature as pf
import findStructure as fs
import calcenergy as ce
import numpy as np

f = []
for i in range(0,100):
    grid = ce.randSurface(5)
    f.append(grid)

test = fv.getFeatureVectors(f)
#test = [fs.findOptimum(5)]
#test2 = fv.getFeatureVectors(test)
pf.plotFeatureMap(test)

