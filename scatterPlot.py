import FeatureVector.featureVector as fv
import FeatureVector.plotFeature as pf
import findStructure as fs
import calcenergy as ce
import numpy as np

f = []
for i in range(0,2):
    grid = ce.randSurface(5)
    f.append(grid)

#test = fv.getBondFeatureVectors(f)
#test = [fs.findOptimum(5)]
test2 = fv.getBondFeatureVectors(f)
#pf.plotFeatureMap(test2)
print(test2)

