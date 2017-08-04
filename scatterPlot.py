import FeatureVector.featureVector as fv
import FeatureVector.plotFeature as pf
import findStructure as fs
import calcenergy as ce
import numpy as np

f = []
for i in range(0,2):
    grid = ce.randSurface(5)
    f.append(grid)

test = fv.getFeatureVectors(f)
pf.plotFeatureMap(test)


